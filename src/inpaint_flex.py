# inpaint_flex.py
import numpy as np
import pandas as pd
import torch
from src.data_utils import prep_df, align_on_date, zscore_apply, zscore_invert
from src.model_unet1d import UNet1D
from src.diffusion import DiffusionConfig, GaussianDiffusion1D

def _dict_to_cfg(d):
    return DiffusionConfig(timesteps=int(d["timesteps"]),
                           beta_schedule=str(d["beta_schedule"]),
                           v_prediction=bool(d["v_prediction"]))

@torch.no_grad()
def inpaint_given_conditioners_from_dfs_flex(
    goog_df, nvda_df, msft_df,
    known_chans,                # e.g., ["returns_MSFT"]
    target_chan="returns_NVDA",
    start_date=None, end_date=None,
    L=84, steps=250, use_ema=True,
    device=None, model_path="ddpm_ts.pt",
    allow_shorten=True
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    cfg = _dict_to_cfg(ckpt["cfg"])
    meta = ckpt["meta"]; mu, sd = meta["mu"], meta["sd"]

    g = prep_df(goog_df, start_date, end_date)
    n = prep_df(nvda_df, start_date, end_date)
    m = prep_df(msft_df, start_date, end_date)
    merged = align_on_date([("GOOG", g), ("NVDA", n), ("MSFT", m)])
    all_channels = [c for c in merged.columns if c.startswith("returns_")]
    train_channels = meta["channels"]  # Only GOOG and MSFT from training
    
    if target_chan not in all_channels:
        raise ValueError(f"target_chan '{target_chan}' not in {all_channels}")
    for kc in known_chans:
        if kc not in train_channels:
            raise ValueError(f"known '{kc}' not in trained channels {train_channels}")

    X_raw = merged[train_channels].astype(float).values
    if X_raw.shape[0] == 0:
        raise ValueError("No rows in selected date range.")
    Xnc = X_raw[:, :, None]
    Xn  = zscore_apply(Xnc, mu, sd)[:, :, 0]

    N = Xn.shape[0]
    used_L = N if (L is None or (N < L and allow_shorten)) else L
    if N < used_L:
        raise ValueError(f"{N} rows < L={L} and allow_shorten=False.")

    window_norm = Xn[-used_L:, :].T
    actual_denorm = zscore_invert(window_norm.T[:, :, None], mu, sd)[:, :, 0].T
    dates = merged["Date"].values[-used_L:]

    C = len(train_channels)  # Only GOOG and MSFT
    mask = np.zeros((C, used_L), dtype=np.float32)
    known_idx = [train_channels.index(kc) for kc in known_chans]
    for i in known_idx:
        mask[i, :] = 1.0

    model = UNet1D(c_in=C, c_out=C, base=64, tdim=128).to(device)
    if use_ema and "state_dict_ema" in ckpt:
        model.load_state_dict(ckpt["state_dict_ema"], strict=True)
    else:
        sd_key = "state_dict_raw" if "state_dict_raw" in ckpt else "state_dict"
        model.load_state_dict(ckpt[sd_key], strict=True)
    diff = GaussianDiffusion1D(model, cfg).to(device)

    # Generate samples from the 2-channel model (GOOG+MSFT)
    known_t = torch.from_numpy(window_norm).float()[None].to(device)
    mask_t  = torch.from_numpy(mask).float()[None].to(device)
    sample_norm = diff.sample_with_mask(known_t, mask_t, steps=steps)[0].cpu().numpy()

    # Denormalize the predicted GOOG+MSFT data
    pred_denorm_2ch = zscore_invert(sample_norm.T[:, :, None], mu, sd)[:, :, 0].T
    
    # Get actual NVDA data for comparison
    nvda_raw = merged["returns_NVDA"].astype(float).values[-used_L:]
    
    # Create full 3-channel output for visualization
    pred_denorm_full = np.zeros((len(all_channels), used_L))
    actual_denorm_full = np.zeros((len(all_channels), used_L))
    
    # Fill in predicted GOOG+MSFT data
    for i, ch in enumerate(train_channels):
        ch_idx = all_channels.index(ch)
        pred_denorm_full[ch_idx] = pred_denorm_2ch[i]
        actual_denorm_full[ch_idx] = actual_denorm[i]
    
    # Add actual NVDA data
    nvda_idx = all_channels.index("returns_NVDA")
    actual_denorm_full[nvda_idx] = nvda_raw
    
    # For prediction, use a simple correlation-based approach as placeholder
    # This is where you'd implement your actual NVDA prediction logic
    goog_pred_idx = train_channels.index("returns_GOOG")
    msft_pred_idx = train_channels.index("returns_MSFT")
    nvda_pred = 0.5 * pred_denorm_2ch[goog_pred_idx] + 0.5 * pred_denorm_2ch[msft_pred_idx]
    pred_denorm_full[nvda_idx] = nvda_pred

    return {
        "channels": all_channels,
        "dates": dates,
        "actual_denorm": actual_denorm_full,
        "pred_denorm": pred_denorm_full,
        "target_index": nvda_idx,
        "known_indices": [all_channels.index(kc) for kc in known_chans],
        "used_L": used_L
    }

