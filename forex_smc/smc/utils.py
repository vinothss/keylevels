from __future__ import annotations

from typing import Optional
import pandas as pd


def to_yahoo_symbol(pair: str) -> str:
    pair = pair.strip().upper()
    if pair.endswith("=X"):
        return pair
    if len(pair) == 6 and pair.isalpha():
        return f"{pair}=X"
    raise ValueError(f"Invalid forex pair format: '{pair}'. Use like 'EURUSD' or 'EURUSD=X'.")


def ohlc_resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if df.empty:
        return df
    agg = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }
    res = df.resample(rule).agg(agg).dropna(how="any")
    res.columns = [c.lower() for c in res.columns]
    return res


def ensure_ohlc_columns(df: pd.DataFrame) -> pd.DataFrame:
    colmap = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "close",
        "Volume": "volume",
    }
    # Lowercase copy
    renamed = {
        (c if c in colmap.values() else colmap.get(c, c)).lower(): df[c]
        for c in df.columns
        if c in df.columns
    }
    out = pd.DataFrame(renamed)
    # Ensure required columns exist
    required = ["open", "high", "low", "close"]
    for r in required:
        if r not in out:
            raise ValueError("Missing required OHLC column: " + r)
    if "volume" not in out:
        out["volume"] = 0.0
    return out


def last_non_nan(series: pd.Series) -> Optional[float]:
    try:
        return series.dropna().iloc[-1]
    except Exception:
        return None