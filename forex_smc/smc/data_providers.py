from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd
import requests
import yfinance as yf

from .utils import ensure_ohlc_columns, ohlc_resample


@dataclass
class YahooConfig:
    interval: str
    period: str


YF_INTERVAL_PERIOD = {
    "15m": YahooConfig(interval="15m", period="60d"),
    "60m": YahooConfig(interval="60m", period="730d"),
    "1d": YahooConfig(interval="1d", period="max"),
    "1wk": YahooConfig(interval="1wk", period="max"),
}


def yahoo_download(symbol: str, key: str) -> pd.DataFrame:
    cfg = YF_INTERVAL_PERIOD[key]
    df = yf.download(symbol, interval=cfg.interval, period=cfg.period, auto_adjust=False, progress=False)
    if isinstance(df, pd.DataFrame) and not df.empty and "Close" in df.columns and df.index.tz is not None:
        df = df.tz_convert("UTC").tz_localize(None)
    return df


def alphavantage_fetch(function: str, from_symbol: str, to_symbol: str, params: Dict[str, str]) -> Optional[dict]:
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        return None
    base = "https://www.alphavantage.co/query"
    query = {"function": function, "from_symbol": from_symbol, "to_symbol": to_symbol, "apikey": api_key, **params}
    resp = requests.get(base, params=query, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict) or any(k.lower().startswith("error") for k in data.keys()):
        return None
    return data


def alphavantage_intraday(from_symbol: str, to_symbol: str, interval: str) -> Optional[pd.DataFrame]:
    data = alphavantage_fetch(
        function="FX_INTRADAY",
        from_symbol=from_symbol,
        to_symbol=to_symbol,
        params={"interval": interval, "outputsize": "full"},
    )
    if not data:
        return None
    key = f"Time Series FX ({interval})"
    ts = data.get(key)
    if not isinstance(ts, dict):
        return None
    df = pd.DataFrame.from_dict(ts, orient="index").sort_index()
    df.index = pd.to_datetime(df.index)
    df.rename(columns={
        "1. open": "open",
        "2. high": "high",
        "3. low": "low",
        "4. close": "close",
    }, inplace=True)
    df = df[["open", "high", "low", "close"]].astype(float)
    df["volume"] = 0.0
    return df


def alphavantage_daily(from_symbol: str, to_symbol: str, func: str) -> Optional[pd.DataFrame]:
    data = alphavantage_fetch(function=func, from_symbol=from_symbol, to_symbol=to_symbol, params={})
    if not data:
        return None
    key = "Time Series FX (Daily)" if func == "FX_DAILY" else "Time Series FX (Weekly)"
    ts = data.get(key)
    if not isinstance(ts, dict):
        return None
    df = pd.DataFrame.from_dict(ts, orient="index").sort_index()
    df.index = pd.to_datetime(df.index)
    df.rename(columns={
        "1. open": "open",
        "2. high": "high",
        "3. low": "low",
        "4. close": "close",
    }, inplace=True)
    df = df[["open", "high", "low", "close"]].astype(float)
    df["volume"] = 0.0
    return df


def get_fx_timeframes(pair: str) -> Dict[str, pd.DataFrame]:
    # pair like EURUSD or EURUSD=X
    # Normalize symbols for AV: split into base/quote
    clean = pair.replace("=X", "").upper()
    if len(clean) != 6:
        raise ValueError(f"Invalid pair: {pair}")
    base, quote = clean[:3], clean[3:]
    yahoo_symbol = f"{clean}=X"

    # Try Yahoo first
    out: Dict[str, pd.DataFrame] = {}
    try:
        df_15m = ensure_ohlc_columns(yahoo_download(yahoo_symbol, "15m"))
        out["15m"] = df_15m
    except Exception:
        out["15m"] = pd.DataFrame()
    try:
        df_60m = yahoo_download(yahoo_symbol, "60m")
        out["4h"] = ohlc_resample(df_60m, "4H")
    except Exception:
        out["4h"] = pd.DataFrame()
    try:
        out["1d"] = ensure_ohlc_columns(yahoo_download(yahoo_symbol, "1d"))
    except Exception:
        out["1d"] = pd.DataFrame()
    try:
        out["1w"] = ensure_ohlc_columns(yahoo_download(yahoo_symbol, "1wk"))
    except Exception:
        out["1w"] = pd.DataFrame()

    # If any are empty, try Alpha Vantage for those
    if out.get("15m", pd.DataFrame()).empty:
        av_15 = alphavantage_intraday(base, quote, "15min")
        if av_15 is not None:
            out["15m"] = av_15
    if out.get("4h", pd.DataFrame()).empty:
        av_60 = alphavantage_intraday(base, quote, "60min")
        if av_60 is not None:
            out["4h"] = ohlc_resample(av_60.rename(columns={c: c.capitalize() for c in av_60.columns}), "4H")
            out["4h"].columns = [c.lower() for c in out["4h"].columns]
    if out.get("1d", pd.DataFrame()).empty:
        av_d = alphavantage_daily(base, quote, "FX_DAILY")
        if av_d is not None:
            out["1d"] = av_d
    if out.get("1w", pd.DataFrame()).empty:
        av_w = alphavantage_daily(base, quote, "FX_WEEKLY")
        if av_w is not None:
            out["1w"] = av_w

    return out