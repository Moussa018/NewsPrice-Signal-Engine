"""Feature engineering shared by training and the feature service.

Keeping this in one place guarantees the model sees identical features
at training time and at serving time.
"""
import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "ret_1", "ret_5", "ret_10",
    "rsi_14", "macd", "macd_signal",
    "ma_ratio_10_50", "vol_10", "volume_ratio",
]

MIN_HISTORY = 60  # bars needed before the last row has valid features


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """df needs columns: close, volume. Returns a frame of FEATURE_COLUMNS."""
    close = df["close"].astype(float)
    volume = df["volume"].astype(float)
    out = pd.DataFrame(index=df.index)

    out["ret_1"] = close.pct_change(1)
    out["ret_5"] = close.pct_change(5)
    out["ret_10"] = close.pct_change(10)

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["rsi_14"] = 100 - 100 / (1 + rs)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    out["macd"] = macd / close
    out["macd_signal"] = macd.ewm(span=9, adjust=False).mean() / close

    out["ma_ratio_10_50"] = close.rolling(10).mean() / close.rolling(50).mean()
    out["vol_10"] = out["ret_1"].rolling(10).std()
    out["volume_ratio"] = volume / volume.rolling(20).mean()

    return out[FEATURE_COLUMNS]
