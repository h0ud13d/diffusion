# train.py
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data_utils import build_windows_from_dfs
from src.model_unet1d import UNet1D
from src.diffusion import DiffusionConfig, GaussianDiffusion1D, EMA

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _cfg_to_dict(cfg: DiffusionConfig):
    return {"timesteps": cfg.timesteps, "beta_schedule": cfg.beta_schedule, "v_prediction": cfg.v_prediction}

def _dict_to_cfg(d):
    return DiffusionConfig(timesteps=int(d["timesteps"]),
                           beta_schedule=str(d["beta_schedule"]),
                           v_prediction=bool(d["v_prediction"]))

def train_ddpm_from_dfs(
    goog_df, nvda_df, msft_df,
    start_date=None, end_date=None,
    L=84, epochs=100, batch=128, steps=1000, lr=2e-4,
    ema_decay=0.999, dropout=0.0, use_amp=True,
    device=None, model_path="ddpm_ts.pt"
):
    set_seed(42)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print(f"Train range: start_date={start_date}, end_date={end_date}")

    tr, va, meta = build_windows_from_dfs(goog_df, nvda_df, msft_df, start_date, end_date, L=L)
    C = len(meta["channels"])

    model = UNet1D(c_in=C, c_out=C, base=64, tdim=128, drop=dropout).to(device)
    cfg = DiffusionConfig(timesteps=steps, beta_schedule="cosine", v_prediction=True)
    diff = GaussianDiffusion1D(model, cfg).to(device)
    ema = EMA(model, decay=ema_decay)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.startswith("cuda")))

    tr_dl = DataLoader(tr, batch_size=batch, shuffle=True, drop_last=True, pin_memory=True)
    va_dl = DataLoader(va, batch_size=batch, shuffle=False, pin_memory=True)

    best = float("inf")
    for ep in range(1, epochs + 1):
        model.train()
        tot = 0.0
        for xb in tr_dl:
            xb = xb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(use_amp and device.startswith("cuda"))):
                loss = diff.loss_on(xb, use_v=True)
            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            ema.update(model)
            tot += loss.item() * xb.size(0)
        tr_loss = tot / len(tr_dl.dataset)

        model.eval()
        with torch.no_grad():
            tot = 0.0
            for xb in va_dl:
                xb = xb.to(device, non_blocking=True)
                loss = diff.loss_on(xb, use_v=True)
                tot += loss.item() * xb.size(0)
            va_loss = tot / len(va_dl.dataset)

        print(f"[{ep:03d}] train {tr_loss:.6f}  val {va_loss:.6f}")

        if va_loss < best:
            best = va_loss
            ema_ckpt = {k: v.clone() for k, v in ema.shadow.items()}
            torch.save({
                "model_cfg": {"c_in": C, "c_out": C, "base": 64, "tdim": 128, "drop": dropout},
                "state_dict_ema": ema_ckpt,
                "state_dict_raw": model.state_dict(),
                "cfg": _cfg_to_dict(cfg),
                "meta": meta,
                "train_range": {"start_date": start_date, "end_date": end_date}
            }, model_path)

    print(f"Saved best model to {model_path}")
    return model_path, meta
