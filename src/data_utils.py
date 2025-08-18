# data_utils.py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# ---------- preprocessing ----------
def ensure_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["Adj Close"].astype(float)
    df["returns"] = np.log(close).diff().fillna(0.0)
    return df

def prep_df(df: pd.DataFrame, start_date=None, end_date=None) -> pd.DataFrame:
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"])
    if start_date is not None:
        out = out[out["Date"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        out = out[out["Date"] <= pd.Timestamp(end_date)]
    out = ensure_returns(out)
    return out

def align_on_date(named_dfs):
    reduced = []
    for sym, d in named_dfs:
        dd = d[["Date", "returns"]].copy()
        dd.rename(columns={"returns": f"returns_{sym}"}, inplace=True)
        reduced.append(dd)
    merged = reduced[0]
    for nxt in reduced[1:]:
        merged = merged.merge(nxt, on="Date", how="inner")
    merged.sort_values("Date", inplace=True)
    merged.reset_index(drop=True, inplace=True)
    return merged

# ---------- normalization ----------
def zscore_fit(x):  # x: (N, C, 1)
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    sd = np.maximum(sd, 1e-8)
    return mu, sd

def zscore_apply(x, mu, sd):
    return (x - mu) / sd

def zscore_invert(x, mu, sd):
    return x * sd + mu

# ---------- dataset ----------
class WindowedDataset(Dataset):
    def __init__(self, X: np.ndarray, L: int):
        self.X = X
        self.L = L
        self.N = X.shape[0]
        self.C = X.shape[1]
        if self.N < self.L:
            raise ValueError(f"Not enough rows ({self.N}) for L={self.L}.")
        self.starts = np.arange(0, self.N - L + 1)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        s = self.starts[i]
        w = self.X[s:s+self.L]    # (L, C)
        return torch.from_numpy(w.T).float()  # (C, L)

def split_dataset(ds, val_frac=0.1):
    n = len(ds.starts)
    n_val = max(1, int(round(n * val_frac)))
    # use earliest (n - n_val) for train, final n_val contiguous windows for val
    tr_idx = np.arange(0, n - n_val)
    va_idx = np.arange(n - n_val, n)
    return (torch.utils.data.Subset(ds, tr_idx),
            torch.utils.data.Subset(ds, va_idx))

# ---------- build windows ----------
def build_windows_from_dfs(goog_df, nvda_df, msft_df, start_date, end_date, L=84):
    from .data_utils import prep_df, align_on_date, zscore_fit, zscore_apply, WindowedDataset, split_dataset
    g = prep_df(goog_df, start_date, end_date)
    n = prep_df(nvda_df, start_date, end_date)
    m = prep_df(msft_df, start_date, end_date)
    merged = align_on_date([("GOOG", g), ("NVDA", n), ("MSFT", m)])

    channels = [c for c in merged.columns if c.startswith("returns_")]
    train_channels = channels  # Include all channels including NVDA
    X_raw = merged[train_channels].astype(float).values      # (N, C) - including all channels
    Xnc = X_raw[:, :, None]                            # (N, C, 1)
    mu, sd = zscore_fit(Xnc)
    Xn = zscore_apply(Xnc, mu, sd)[:, :, 0]            # (N, C)

    ds = WindowedDataset(Xn, L=L)
    tr, va = split_dataset(ds, val_frac=0.1)
    meta = {
        "mu": mu, "sd": sd, "channels": train_channels,
        "dates": merged["Date"].values, "all_channels": channels
    }
    return tr, va, meta

