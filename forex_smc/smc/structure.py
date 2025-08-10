from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd


def detect_swings(df: pd.DataFrame, left: int = 2, right: int = 2) -> pd.DataFrame:
    out = df.copy()
    highs = out["high"].values
    lows = out["low"].values
    n = len(out)
    swing_high = np.zeros(n, dtype=bool)
    swing_low = np.zeros(n, dtype=bool)

    for i in range(left, n - right):
        if highs[i] == np.max(highs[i - left : i + right + 1]):
            # Strictly greater than neighbors
            if highs[i] > np.max(highs[i - left : i]) and highs[i] >= np.max(highs[i + 1 : i + right + 1]):
                swing_high[i] = True
        if lows[i] == np.min(lows[i - left : i + right + 1]):
            if lows[i] < np.min(lows[i - left : i]) and lows[i] <= np.min(lows[i + 1 : i + right + 1]):
                swing_low[i] = True

    out["swing_high"] = swing_high
    out["swing_low"] = swing_low
    return out


def detect_structure_events(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    highs = out["high"].values
    lows = out["low"].values
    closes = out["close"].values

    swing_high_idx = list(np.where(out["swing_high"].values)[0])
    swing_low_idx = list(np.where(out["swing_low"].values)[0])

    bos_up = np.zeros(len(out), dtype=bool)
    bos_down = np.zeros(len(out), dtype=bool)
    choch_up = np.zeros(len(out), dtype=bool)
    choch_down = np.zeros(len(out), dtype=bool)

    last_direction: str | None = None  # 'up' or 'down'
    last_bos_dir: str | None = None

    # Maintain latest unbroken swing levels
    pending_highs = []  # (idx, price)
    pending_lows = []   # (idx, price)

    for i in range(len(out)):
        # Append any newly confirmed swing levels up to i
        if i in swing_high_idx:
            pending_highs.append((i, highs[i]))
        if i in swing_low_idx:
            pending_lows.append((i, lows[i]))

        # Remove levels that are in the far past to keep lists compact (optional)
        # Keep last ~500 bars worth
        while pending_highs and pending_highs[0][0] < i - 2000:
            pending_highs.pop(0)
        while pending_lows and pending_lows[0][0] < i - 2000:
            pending_lows.pop(0)

        # Check for breaks by close
        if pending_highs:
            last_high_idx, last_high_price = pending_highs[-1]
            if closes[i] > last_high_price:
                bos_up[i] = True
                if last_direction == "down":
                    choch_up[i] = True
                last_direction = "up"
                last_bos_dir = "up"
                # Once broken, that level is no longer pending
                pending_highs = [ph for ph in pending_highs if ph[0] > last_high_idx]
        if pending_lows:
            last_low_idx, last_low_price = pending_lows[-1]
            if closes[i] < last_low_price:
                bos_down[i] = True
                if last_direction == "up":
                    choch_down[i] = True
                last_direction = "down"
                last_bos_dir = "down"
                pending_lows = [pl for pl in pending_lows if pl[0] > last_low_idx]

    out["bos_up"] = bos_up
    out["bos_down"] = bos_down
    out["choch_up"] = choch_up
    out["choch_down"] = choch_down

    trend = pd.Series(index=out.index, dtype="object")
    trend[(choch_up | bos_up)] = "up"
    trend[(choch_down | bos_down)] = "down"
    out["structure_trend"] = trend
    return out


def annotate_structure(df: pd.DataFrame, swing_left: int = 2, swing_right: int = 2) -> pd.DataFrame:
    df1 = detect_swings(df, swing_left, swing_right)
    df2 = detect_structure_events(df1)
    return df2