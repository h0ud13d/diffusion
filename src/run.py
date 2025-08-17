from src.train import train_ddpm_from_dfs
from src.inpaint_flex import inpaint_given_conditioners_from_dfs_flex
from src.graph import output

import pandas as pd

goog_df = pd.read_csv("stocks/GOOG.csv")
nvda_df = pd.read_csv("stocks/NVDA.csv")
msft_df = pd.read_csv("stocks/MSFT.csv")

model_path, meta = train_ddpm_from_dfs(
    goog_df, nvda_df, msft_df,
    start_date="2018-01-01", end_date=None,
    L=84, epochs=150, batch=128, steps=1000, lr=2e-4,
    ema_decay=0.999, dropout=0.0, use_amp=True,
    model_path="ddpm_ts.pt"
)

res = inpaint_given_conditioners_from_dfs_flex(
    goog_df, nvda_df, msft_df,
    known_chans=["returns_GOOG", "returns_MSFT"], target_chan="returns_NVDA",
    start_date="2020-01-01", end_date="2020-03-01",
    L=84, steps=250, use_ema=True, allow_shorten=True, model_path="ddpm_ts.pt"
)

output(res)
