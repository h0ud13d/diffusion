import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb, to_hex

import numpy as np
import pandas as pd

def _lighten(color: str, amount: float = 0.4) -> str:
    """Blend color toward white by `amount` in [0,1]."""
    c = np.array(to_rgb(color))
    w = np.array([1.0, 1.0, 1.0])
    return to_hex((1 - amount) * c + amount * w)

def output(res):
    dates = pd.to_datetime(res["dates"])
    channels = res["channels"]
    A = res["actual_denorm"]  # (C, T)
    P = res["pred_denorm"]    # (C, T)

    idx_nvda = channels.index("returns_NVDA")
    idx_msft = channels.index("returns_MSFT")

    nvda_actual = A[idx_nvda]
    nvda_pred   = P[idx_nvda]
    msft_actual = A[idx_msft]

    nvda_color = "#FFA500"   # 
    msft_color = "#0000FF"   #
    nvda_pred_color = _lighten(nvda_color, amount=0.45)

    plt.figure(figsize=(14, 6))
    plt.plot(dates, nvda_actual, label="NVDA Actual", linewidth=2.2, color=nvda_color)
    plt.plot(dates, nvda_pred,   label="NVDA Inpainted", linewidth=2.2, linestyle="--", color=nvda_pred_color)

    # MSFT: actual
    plt.plot(dates, msft_actual, label="MSFT Actual", linewidth=1.6, color=msft_color, alpha=0.9)

    plt.axhline(0, linewidth=0.9, linestyle=":", color="#222222", alpha=0.6)
    plt.xlabel("Date"); plt.ylabel("Daily Returns")
    plt.legend(ncols=2)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig("imgs/nvda_msft_inpaint.png", dpi=150)
    plt.show()

