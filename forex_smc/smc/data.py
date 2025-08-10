from __future__ import annotations

from typing import Dict

import pandas as pd

from .data_providers import get_fx_timeframes


SUPPORTED_TFS = ["15m", "4h", "1d", "1w"]


def get_timeframes_data(pair: str) -> Dict[str, pd.DataFrame]:
    return get_fx_timeframes(pair)