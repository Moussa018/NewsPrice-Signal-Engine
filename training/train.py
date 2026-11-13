"""Train the price-direction model: will tomorrow's close be above today's?

Run from the project root:  python training/train.py
"""
import sys
from pathlib import Path

import pandas as pd
import xgboost as xgb
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "shared"))
from features import FEATURE_COLUMNS, compute_features  # noqa: E402

SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "BTC-USD", "ETH-USD"]
OUT = ROOT / "models" / "model.json"


def load(symbol: str) -> pd.DataFrame:
    raw = yf.download(symbol, period="10y", interval="1d", auto_adjust=True, progress=False)
    raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
    df = raw.rename(columns=str.lower)[["close", "volume"]].dropna()
    feats = compute_features(df)
    feats["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    feats.loc[df["close"].shift(-1).isna(), "target"] = pd.NA  # last row has no label
    feats["symbol"] = symbol
    return feats.dropna()


def main() -> None:
    data = pd.concat([load(s) for s in SYMBOLS]).sort_index()
    print(f"rows: {len(data)}")

    # Time-based split: never test on data older than the training data.
    cutoff = data.index[int(len(data) * 0.8)]
    train, test = data[data.index < cutoff], data[data.index >= cutoff]
    print(f"train until {cutoff.date()}  train={len(train)}  test={len(test)}")

    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
    )
    model.fit(train[FEATURE_COLUMNS], train["target"].astype(int))

    acc = (model.predict(test[FEATURE_COLUMNS]) == test["target"].astype(int)).mean()
    baseline = max(test["target"].mean(), 1 - test["target"].mean())
    print(f"test accuracy: {acc:.3f}   always-guess-majority baseline: {baseline:.3f}")

    OUT.parent.mkdir(exist_ok=True)
    model.save_model(OUT)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
