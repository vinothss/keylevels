from __future__ import annotations

import numpy as np
import pandas as pd


def detect_fvg(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    high = out["high"].values
    low = out["low"].values
    n = len(out)

    bull_low = np.full(n, np.nan)
    bull_high = np.full(n, np.nan)
    bear_low = np.full(n, np.nan)
    bear_high = np.full(n, np.nan)

    # Use classic 3-candle definition: candle i forms FVG with i-2
    for i in range(2, n):
        # Bullish FVG when low[i] > high[i-2]
        if low[i] > high[i - 2]:
            bull_low[i] = high[i - 2]
            bull_high[i] = low[i]
        # Bearish FVG when high[i] < low[i-2]
        if high[i] < low[i - 2]:
            bear_low[i] = high[i]
            bear_high[i] = low[i - 2]

    out["fvg_bull_low"] = bull_low
    out["fvg_bull_high"] = bull_high
    out["fvg_bear_low"] = bear_low
    out["fvg_bear_high"] = bear_high
    return out


def latest_fvgs(df: pd.DataFrame, lookback: int = 500) -> dict:
    tail = df.tail(lookback)
    bulls = tail[(~tail["fvg_bull_low"].isna()) & (~tail["fvg_bull_high"].isna())]
    bears = tail[(~tail["fvg_bear_low"].isna()) & (~tail["fvg_bear_high"].isna())]

    def rows_to_list(frame: pd.DataFrame, low_col: str, high_col: str):
        return [
            {
                "timestamp": str(idx),
                "low": float(row[low_col]),
                "high": float(row[high_col]),
            }
            for idx, row in frame[[low_col, high_col]].iterrows()
        ]

    return {
        "bullish": rows_to_list(bulls, "fvg_bull_low", "fvg_bull_high"),
        "bearish": rows_to_list(bears, "fvg_bear_low", "fvg_bear_high"),
    }