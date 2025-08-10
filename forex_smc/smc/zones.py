from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd


def _latest_enclosing_range(df: pd.DataFrame) -> Optional[Tuple[int, float, int, float]]:
    swing_high_idx = np.where(df.get("swing_high", pd.Series(False, index=df.index)).values)[0]
    swing_low_idx = np.where(df.get("swing_low", pd.Series(False, index=df.index)).values)[0]
    if len(swing_high_idx) == 0 or len(swing_low_idx) == 0:
        return None
    last_idx = len(df) - 1

    # Find the nearest swing high and swing low before the last index
    last_sh_i = swing_high_idx[swing_high_idx <= last_idx][-1] if np.any(swing_high_idx <= last_idx) else swing_high_idx[-1]
    last_sl_i = swing_low_idx[swing_low_idx <= last_idx][-1] if np.any(swing_low_idx <= last_idx) else swing_low_idx[-1]

    high_price = float(df.iloc[last_sh_i]["high"]) if last_sh_i is not None else None
    low_price = float(df.iloc[last_sl_i]["low"]) if last_sl_i is not None else None
    if high_price is None or low_price is None:
        return None

    # Ensure the range is oriented high vs low correctly
    if last_sh_i > last_sl_i:
        # High came after low: use last swing low before the high
        prev_lows = [i for i in swing_low_idx if i < last_sh_i]
        if prev_lows:
            last_sl_i = prev_lows[-1]
            low_price = float(df.iloc[last_sl_i]["low"])
    else:
        # Low came after high: use last swing high before the low
        prev_highs = [i for i in swing_high_idx if i < last_sl_i]
        if prev_highs:
            last_sh_i = prev_highs[-1]
            high_price = float(df.iloc[last_sh_i]["high"])

    if high_price is None or low_price is None:
        return None

    return last_sh_i, high_price, last_sl_i, low_price


def compute_zones(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    res = _latest_enclosing_range(out)
    if res is None:
        out["eq_level"] = np.nan
        out["range_high"] = np.nan
        out["range_low"] = np.nan
        out["zone"] = np.nan
        return out

    hi_idx, hi, lo_idx, lo = res
    eq = (hi + lo) / 2.0
    out["eq_level"] = eq
    out["range_high"] = hi
    out["range_low"] = lo

    # Classify last close
    last_close = float(out.iloc[-1]["close"]) if len(out) else np.nan
    if np.isnan(last_close):
        out["zone"] = np.nan
        return out

    # Tolerate tiny distances as equilibrium
    tol = max((hi - lo) * 0.002, 1e-8)
    if abs(last_close - eq) <= tol:
        zone = "equilibrium"
    elif last_close > eq:
        zone = "premium"
    else:
        zone = "discount"
    out["zone"] = zone
    return out


def summarize_zone_levels(df: pd.DataFrame) -> dict:
    last = df.iloc[-1]
    return {
        "range_high": float(last.get("range_high", np.nan)),
        "range_low": float(last.get("range_low", np.nan)),
        "equilibrium": float(last.get("eq_level", np.nan)),
        "zone": last.get("zone", None),
    }