#!/usr/bin/env python3
"""
Diffusion for Time‑Series: conditional DDPM that generates a distribution of future returns
conditioned on a history window. Minimal, readable, and trainable on 1 GPU/CPU for demos.

Usage (quick demo with yfinance):
  python diffusion.py --ticker AAPL --download --epochs 10 --horizon 20 --context 64

Usage (CSV):
  python diffusion.py --csv path/to/data.csv --price-col Close --date-col Date \
      --epochs 20 --horizon 20 --context 64

The model learns to generate H future log‑returns given the last L context returns.
You can turn samples into price paths and compute predictive intervals & trading signals.

Notes:
- This is a pedagogical baseline. For serious work consider architectures with attention,
  improved schedules, Denoising Diffusion Implicit Models (DDIM), or SDE-based training.
- Beware of data leakage, non‑stationarity, and overfitting. Use walk‑forward splits.
"""

import argparse
import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    import yfinance as yf  # optional
except Exception:
    yf = None

# -----------------------------
# Utilities
# -----------------------------

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)


def to_device(x):
    return x.cuda() if torch.cuda.is_available() else x


# -----------------------------
# Data
# -----------------------------

class ReturnsDataset(Dataset):
    def __init__(self, returns: np.ndarray, L: int, H: int, start: int, end: int):
        super().__init__()
        self.r = returns.astype(np.float32)
        self.L, self.H = L, H
        self.idx = []
        # i indexes the start of the forecast window [i, i+H)
        for i in range(max(L, start), min(len(self.r) - H, end)):
            self.idx.append(i)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        i = self.idx[k]
        ctx = self.r[i - self.L : i]           # shape [L]
        fut = self.r[i : i + self.H]           # shape [H]
        if ctx.shape[0] != self.L:
            raise RuntimeError(f"Context length mismatch: expected L={self.L}, got {ctx.shape[0]}.")
        if fut.shape[0] != self.H:
            raise RuntimeError(f"Horizon length mismatch: expected H={self.H}, got {fut.shape[0]}.")
        return torch.from_numpy(ctx), torch.from_numpy(fut)


@dataclass
class Split:
    train_start: int
    train_end: int
    val_start: int
    val_end: int
    test_start: int
    test_end: int


def make_walk_forward_splits(N: int, val_frac=0.15, test_frac=0.15) -> Split:
    test_len = int(N * test_frac)
    val_len = int(N * val_frac)
    train_end = N - test_len - val_len
    return Split(0, train_end, train_end, train_end + val_len, train_end + val_len, N)


# -----------------------------
# Beta schedule & diffusion helpers (DDPM, epsilon‑prediction)
# -----------------------------

def make_beta_schedule(T: int, beta_start=1e-4, beta_end=2e-2):
    return torch.linspace(beta_start, beta_end, T)


class Diffusion(nn.Module):
    def __init__(self, T: int = 1000, beta_start=1e-4, beta_end=2e-2):
        super().__init__()
        beta = make_beta_schedule(T, beta_start, beta_end)
        alpha = 1.0 - beta
        alphabar = torch.cumprod(alpha, dim=0)
        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alphabar", alphabar)
        self.T = T

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None):
        # x0: [B, 1, H]
        if eps is None:
            eps = torch.randn_like(x0)
        a_bar = self.alphabar[t].view(-1, 1, 1)
        noisy = torch.sqrt(a_bar) * x0 + torch.sqrt(1 - a_bar) * eps
        return noisy, eps


# -----------------------------
# Time & conditioning embeddings
# -----------------------------

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t: [B] int timesteps; map to [B, dim]
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, half, device=device).float() / (half - 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0,1))
        return emb


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(), nn.Linear(hidden, out_dim)
        )
    def forward(self, x):
        return self.net(x)


# -----------------------------
# Tiny 1D U‑Net with FiLM conditioning from context & time
# -----------------------------

class FiLM(nn.Module):
    def __init__(self, emb_dim, channels):
        super().__init__()
        self.to_scale = nn.Linear(emb_dim, channels)
        self.to_shift = nn.Linear(emb_dim, channels)
    def forward(self, x, emb):
        # x: [B, C, L], emb: [B, E]
        s = self.to_scale(emb).unsqueeze(-1)
        b = self.to_shift(emb).unsqueeze(-1)
        return x * (1 + s) + b


class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.gn1 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.gn2 = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU()
        self.film = FiLM(emb_dim, out_ch)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    def forward(self, x, emb):
        h = self.conv1(x)
        h = self.gn1(h)
        h = self.act(h)
        h = self.film(h, emb)
        h = self.conv2(h)
        h = self.gn2(h)
        return self.act(h + self.skip(x))


class TinyUNet1D(nn.Module):
    def __init__(self, horizon: int, ctx_len: int, base_ch: int = 64, time_dim: int = 128):
        super().__init__()
        self.horizon = horizon
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(time_dim), nn.Linear(time_dim, time_dim), nn.SiLU()
        )
        self.ctx_emb = MLP(ctx_len, time_dim)
        # encoder
        self.rb1 = ResBlock1D(1, base_ch, time_dim * 2)
        self.down = nn.Conv1d(base_ch, base_ch, 4, stride=2, padding=1)
        self.rb2 = ResBlock1D(base_ch, base_ch * 2, time_dim * 2)
        # bottleneck
        self.rb3 = ResBlock1D(base_ch * 2, base_ch * 2, time_dim * 2)
        # decoder
        self.up = nn.ConvTranspose1d(base_ch * 2, base_ch, 4, stride=2, padding=1)
        self.rb4 = ResBlock1D(base_ch * 2, base_ch, time_dim * 2)
        self.out = nn.Conv1d(base_ch, 1, 1)

    def forward(self, x, t, ctx):
        # x: [B,1,H], t:[B], ctx:[B,L]
        te = self.time_emb(t)
        ce = self.ctx_emb(ctx)
        emb = torch.cat([te, ce], dim=-1)  # [B, 2*time_dim]
        
        h1 = self.rb1(x, emb)
        d = self.down(h1)
        h2 = self.rb2(d, emb)
        b = self.rb3(h2, emb)
        u = self.up(b)
        u = torch.cat([u, h1], dim=1)
        h3 = self.rb4(u, emb)
        return self.out(h3)


# -----------------------------
# Training & sampling
# -----------------------------

class Model(nn.Module):
    def __init__(self, horizon: int, ctx_len: int, T: int = 1000):
        super().__init__()
        self.diff = Diffusion(T=T)
        self.net = TinyUNet1D(horizon=horizon, ctx_len=ctx_len)

    def forward(self, noisy_x, t, ctx):
        return self.net(noisy_x, t, ctx)

    @torch.no_grad()
    def sample(self, B: int, ctx: torch.Tensor) -> torch.Tensor:
        device = next(self.parameters()).device
        H = self.net.horizon
        x = torch.randn(B, 1, H, device=device)
        for t in reversed(range(self.diff.T)):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            eps = self.net(x, t_batch, ctx)
            beta_t = self.diff.beta[t]
            alpha_t = self.diff.alpha[t]
            a_bar_t = self.diff.alphabar[t]
            # DDPM posterior mean
            mean = (1 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1 - a_bar_t)) * eps)
            if t > 0:
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(beta_t) * noise
            else:
                x = mean
        return x.squeeze(1)  # [B, H]


# -----------------------------
# Metrics
# -----------------------------

def directional_accuracy(samples: torch.Tensor, targets: torch.Tensor) -> float:
    # samples: [B,S,H], targets: [B,H]
    mean_path = samples.mean(dim=1)
    pred_up = (mean_path.sum(dim=1) > 0).float()
    true_up = (targets.sum(dim=1) > 0).float()
    return (pred_up == true_up).float().mean().item()


def crps_approx(samples: torch.Tensor, targets: torch.Tensor) -> float:
    # scalar CRPS per step averaged; samples [B,S,H], targets [B,H]
    # approximation using sample CDF at target values
    B, S, H = samples.shape
    t = targets.unsqueeze(1).expand(-1, S, -1)
    # CRPS = E|X - y| - 0.5 E|X - X'|
    term1 = (samples - t).abs().mean(dim=1)
    term2 = 0.5 * (samples.unsqueeze(2) - samples.unsqueeze(1)).abs().mean(dim=(1,2))
    return (term1.mean() - term2.mean()).item()


# -----------------------------
# Training loop
# -----------------------------

def train(model: Model, loader: DataLoader, val_loader: DataLoader, epochs: int, lr: float, grad_clip: float, outdir: str):
    device = next(model.parameters()).device
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    best_val = float('inf')
    os.makedirs(outdir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for ctx, fut in loader:
            ctx = to_device(ctx)
            fut = to_device(fut).unsqueeze(1)  # [B,1,H]
            B = fut.size(0)
            t = torch.randint(0, model.diff.T, (B,), device=ctx.device, dtype=torch.long)
            x_t, eps = model.diff.add_noise(fut, t)
            eps_pred = model(x_t, t, ctx)
            loss = F.mse_loss(eps_pred, eps)
            opt.zero_grad()
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            total += loss.item() * B
        train_loss = total / len(loader.dataset)
        val_loss = evaluate_mse(model, val_loader)
        print(f"Epoch {epoch:03d} | train MSE {train_loss:.5f} | val MSE {val_loss:.5f}")
        torch.save(model.state_dict(), os.path.join(outdir, "last.pt"))
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(outdir, "best.pt"))


def evaluate_mse(model: Model, loader: DataLoader) -> float:
    device = next(model.parameters()).device
    model.eval()
    total = 0.0
    with torch.no_grad():
        for ctx, fut in loader:
            ctx = to_device(ctx)
            fut = to_device(fut).unsqueeze(1)
            B = fut.size(0)
            t = torch.randint(0, model.diff.T, (B,), device=ctx.device, dtype=torch.long)
            x_t, eps = model.diff.add_noise(fut, t)
            eps_pred = model(x_t, t, ctx)
            loss = F.mse_loss(eps_pred, eps)
            total += loss.item() * B
    return total / len(loader.dataset)


# -----------------------------
# Data ingestion & preprocessing
# -----------------------------

def load_prices(args) -> pd.DataFrame:
    if args.csv:
        df = pd.read_csv(args.csv)
        if args.date_col and args.date_col in df.columns:
            df[args.date_col] = pd.to_datetime(df[args.date_col])
            df = df.sort_values(args.date_col)
        return df
    if args.download:
        if yf is None:
            raise RuntimeError("yfinance not installed; provide --csv instead or pip install yfinance")
        if args.start is None:
            args.start = "2005-01-01"
        # First attempt: explicit start, no auto-adjust surprises
        df = yf.download(args.ticker, start=args.start, progress=False, auto_adjust=False)
        if df is None or df.empty:
            # Fallback: pull max period with auto_adjust to improve availability
            df = yf.download(args.ticker, period="max", progress=False, auto_adjust=True)
        if df is None or df.empty:
            raise ValueError(f"No price data returned for {args.ticker}. Try a different --start or check ticker.")
        df = df.reset_index()
        return df
    raise RuntimeError("Provide --csv or --download with --ticker")


def compute_returns(df: pd.DataFrame, price_col: str) -> np.ndarray:
    # Try to find a usable price column: prefer provided, then Adj Close, then Close
    candidates = [price_col]
    for alt in ["Adj Close", "Close", "adjclose", "close"]:
        if alt not in candidates:
            candidates.append(alt)
    series = None
    for c in candidates:
        # Handle MultiIndex columns
        matching_cols = [col for col in df.columns if (isinstance(col, tuple) and col[0] == c) or col == c]
        if matching_cols:
            col = matching_cols[0]
            s = pd.to_numeric(df[col], errors='coerce')
            s = s.replace([np.inf, -np.inf], np.nan).dropna()
            s = s[s > 0]  # log requires positive
            if s.size >= 2:
                series = s
                break
    if series is None or series.size < 2:
        raise ValueError(
            f"Could not find a usable price column among {candidates}. "
            f"Try --price-col 'Adj Close' or check the downloaded data."
        )
    price = series.to_numpy()
    r = np.diff(np.log(price))  # log returns
    return r


def standardize_train_only(r: np.ndarray, split: Split) -> Tuple[np.ndarray, float, float]:
    mu = r[split.train_start : split.train_end].mean()
    sd = r[split.train_start : split.train_end].std() + 1e-8
    return (r - mu) / sd, mu, sd


# -----------------------------
# Backtest helpers
# -----------------------------

def backtest_directional(model: Model, loader: DataLoader, samples: int = 50) -> Tuple[float, float]:
    model.eval()
    all_samps = []
    all_t = []
    with torch.no_grad():
        for ctx, fut in loader:
            ctx = to_device(ctx)
            fut = to_device(fut)
            B = ctx.size(0)
            ctx_rep = ctx
            samp = []
            for _ in range(samples):
                s = model.sample(B, ctx_rep)  # [B,H]
                samp.append(s)
            samp = torch.stack(samp, dim=1)  # [B,S,H]
            all_samps.append(samp.cpu())
            all_t.append(fut.cpu())
    S = torch.cat(all_samps, dim=0)
    Tt = torch.cat(all_t, dim=0)
    da = directional_accuracy(S, Tt)
    crps = crps_approx(S, Tt)
    return da, crps


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    src = parser.add_mutually_exclusive_group()
    src.add_argument('--csv', type=str, default=None, help='CSV with prices')
    src.add_argument('--download', action='store_true')
    parser.add_argument('--ticker', type=str, default='AAPL')
    parser.add_argument('--start', type=str, default=None)
    parser.add_argument('--price-col', type=str, default='Close')
    parser.add_argument('--date-col', type=str, default='Date')
    parser.add_argument('--context', type=int, default=64, help='L: context length')
    parser.add_argument('--horizon', type=int, default=20, help='H: forecast horizon (days)')
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--T', type=int, default=1000, help='diffusion steps')
    parser.add_argument('--outdir', type=str, default='runs/tsdiff')
    args = parser.parse_args()

    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    df = load_prices(args)
    if args.price_col not in df.columns:
        print(f"Warning: price column '{args.price_col}' not found; will try fallbacks like 'Adj Close'/'Close'.")

    # Debug diagnostics for data quality
    cols = df.columns.tolist()
    def col_stat(c):
        # Handle MultiIndex columns
        matching_cols = [col for col in df.columns if (isinstance(col, tuple) and col[0] == c) or col == c]
        if matching_cols:
            col = matching_cols[0]
            s = pd.to_numeric(df[col], errors='coerce')
            return f"{c}: n={s.shape[0]}, NaNs={s.isna().sum()}"
        return f"{c}: missing"
    print("Downloaded rows:", len(df), "| columns:", cols)
    print("Price diagnostics:", col_stat(args.price_col), "|", col_stat('Adj Close'), "|", col_stat('Close'))

    r = compute_returns(df, args.price_col)
    r = r[~np.isnan(r)]
    N = len(r)

    # Validate minimum length
    L, H = args.context, args.horizon
    min_required = L + H + 1
    if N < min_required:
        raise ValueError(
            f"Not enough data after preprocessing: returns N={N}, need at least {min_required} (context {L} + horizon {H} + 1). "
            f"Try earlier --start, smaller --context/--horizon, or use --price-col 'Adj Close'."
        )

    split = make_walk_forward_splits(N)
    r_std, mu, sd = standardize_train_only(r, split)

    train_ds = ReturnsDataset(r_std, L, H, split.train_start, split.train_end)
    val_ds   = ReturnsDataset(r_std, L, H, split.val_start,   split.val_end)
    test_ds  = ReturnsDataset(r_std, L, H, split.test_start,  split.test_end)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=args.batch, shuffle=False)

    model = Model(horizon=H, ctx_len=L, T=args.T).to(device)

    print(f"Train {len(train_ds)} | Val {len(val_ds)} | Test {len(test_ds)} | Device {device}")

    train(model, train_loader, val_loader, args.epochs, args.lr, args.grad_clip, args.outdir)

    # Load best and evaluate probabilistically
    model.load_state_dict(torch.load(os.path.join(args.outdir, 'best.pt'), map_location=device))
    da, crps = backtest_directional(model, test_loader, samples=64)
    print(f"Test Directional Accuracy (sum over horizon): {da:.3f} | CRPS: {crps:.6f}")

    # Sample a few future paths from the last available context
    if len(test_ds) > 0:
        ctx_last, _ = test_ds[-1]
        ctx_last = ctx_last.unsqueeze(0).to(device)
        with torch.no_grad():
            paths = []
            for _ in range(20):
                s = model.sample(1, ctx_last)  # [1,H]
                paths.append(s.cpu().numpy()[0])
            paths = np.stack(paths, axis=0)
        # Convert returns back to price paths starting from last observed price
        # Prefer Adj Close if available for a more stable base
        last_price_col = 'Adj Close' if 'Adj Close' in df.columns else (args.price_col if args.price_col in df.columns else 'Close')
        last_price_series = pd.to_numeric(df[last_price_col], errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
        last_price = float(last_price_series.to_numpy()[-1])

        mean_path = paths.mean(axis=0)
        q10 = np.quantile(paths, 0.10, axis=0)
        q90 = np.quantile(paths, 0.90, axis=0)
        price_mean = last_price * np.exp(np.cumsum(mean_path * sd + mu))
        price_q10  = last_price * np.exp(np.cumsum(q10 * sd + mu))
        price_q90  = last_price * np.exp(np.cumsum(q90 * sd + mu))
        out = pd.DataFrame({
            't+': np.arange(1, len(price_mean)+1),
            'price_mean': price_mean,
            'price_q10': price_q10,
            'price_q90': price_q90,
        })
        outpath = os.path.join(args.outdir, 'forecast.csv')
        os.makedirs(args.outdir, exist_ok=True)
        out.to_csv(outpath, index=False)
        print(f"Saved forecast quantiles to {outpath}")


if __name__ == '__main__':
    main()
