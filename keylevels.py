import argparse
from collections import Counter
from datetime import datetime
import time
from typing import Dict, List, Tuple

import pandas as pd
import yfinance as yf

try:
    import schedule  # type: ignore
except Exception:  # pragma: no cover
    schedule = None  # If schedule is not installed, users can't use --schedule


# ---------------------------
# DATA DOWNLOAD
# ---------------------------
def download_data(
    ticker: str,
    start: str = "2024-01-01",
    end: str | None = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """Download OHLCV data for *ticker* using yfinance."""
    df = yf.download(ticker, start=start, end=end, interval=interval, group_by="ticker")
    df.dropna(inplace=True)
    df.index = pd.to_datetime(df.index)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(1)
    return df


# ---------------------------
# SWING DETECTION
# ---------------------------
def find_swing_points(df: pd.DataFrame, window: int = 3, price_decimals: int = 2) -> pd.DataFrame:
    """Identify swing points (HH, LL, HL, LH) in the price data."""
    swings: List[Tuple[pd.Timestamp, str, float]] = []
    for i in range(window, len(df) - window):
        high = df["High"].iloc[i]
        low = df["Low"].iloc[i]

        prev_highs = df["High"].iloc[i - window : i].values
        next_highs = df["High"].iloc[i + 1 : i + 1 + window].values
        prev_lows = df["Low"].iloc[i - window : i].values
        next_lows = df["Low"].iloc[i + 1 : i + 1 + window].values

        is_hh = high > max(prev_highs) and high > max(next_highs)
        is_ll = low < min(prev_lows) and low < min(next_lows)
        is_hl = (
            low > df["Low"].iloc[i - 1]
            and low > df["Low"].iloc[i - 2]
            and low < df["Low"].iloc[i + 1]
            and low < df["Low"].iloc[i + 2]
        )
        is_lh = (
            high < df["High"].iloc[i - 1]
            and high < df["High"].iloc[i - 2]
            and high > df["High"].iloc[i + 1]
            and high > df["High"].iloc[i + 2]
        )

        if is_hh:
            swings.append((df.index[i], "HH", round(high, price_decimals)))
        elif is_ll:
            swings.append((df.index[i], "LL", round(low, price_decimals)))
        elif is_hl:
            swings.append((df.index[i], "HL", round(low, price_decimals)))
        elif is_lh:
            swings.append((df.index[i], "LH", round(high, price_decimals)))

    return pd.DataFrame(swings, columns=["Date", "Type", "Price"])


# ---------------------------
# KEY LEVEL AGGREGATION
# ---------------------------
def get_key_levels(swings_df: pd.DataFrame, min_count: int = 2) -> Dict[float, Dict[str, int]]:
    """Aggregate swing points into key price levels."""
    price_counts = Counter(swings_df["Price"])
    price_roles: Dict[float, Dict[str, int]] = {}

    for price in price_counts:
        types = swings_df[swings_df["Price"] == price]["Type"].tolist()
        if types.count("LL") + types.count("HL") > types.count("HH") + types.count("LH"):
            role = "Support"
        elif types.count("HH") + types.count("LH") > types.count("LL") + types.count("HL"):
            role = "Resistance"
        else:
            role = "Both"

        price_roles[price] = {"count": price_counts[price], "role": role}

    return {
        price: data
        for price, data in price_roles.items()
        if data["count"] >= min_count
    }


def backtest_key_levels(
    df: pd.DataFrame,
    levels: Dict[float, Dict[str, int]],
    lookahead: int = 5,
) -> Dict[float, Dict[str, float]]:
    """Backtest *levels* against historical price data.

    For each level, every candle that touches it is examined to determine
    whether price subsequently "bounces" away or "breaks" through within the
    next *lookahead* candles. Statistics for touches, bounce ratio and
    average moves after bounces/breaks are returned for later reporting.

    Parameters
    ----------
    df:
        Price data with columns ``High`` and ``Low`` (and ``Close`` for
        direction detection).
    levels:
        Levels as produced by :func:`get_key_levels`.
    lookahead:
        Number of candles to inspect after each touch.

    Returns
    -------
    Dict[float, Dict[str, float]]
        Mapping of level price to backtest statistics.
    """

    stats: Dict[float, Dict[str, float]] = {}
    for level, info in levels.items():
        touches = bounces = breaks = 0
        move_bounce: List[float] = []
        move_break: List[float] = []

        for i in range(1, len(df) - lookahead):
            high = df["High"].iloc[i]
            low = df["Low"].iloc[i]

            if low <= level <= high:
                touches += 1
                prev_close = df["Close"].iloc[i - 1]
                future = df.iloc[i + 1 : i + 1 + lookahead]

                if prev_close > level:  # approached from above -> support test
                    if future["Low"].min() < level:
                        breaks += 1
                        move_break.append(level - future["Low"].min())
                    else:
                        bounces += 1
                        move_bounce.append(future["High"].max() - level)
                else:  # approached from below -> resistance test
                    if future["High"].max() > level:
                        breaks += 1
                        move_break.append(future["High"].max() - level)
                    else:
                        bounces += 1
                        move_bounce.append(level - future["Low"].min())

        if touches:
            stats[level] = {
                "touches": float(touches),
                "bounces": float(bounces),
                "breaks": float(breaks),
                "bounce_ratio": bounces / touches if touches else 0.0,
                "avg_move_bounce": (
                    sum(move_bounce) / len(move_bounce) if move_bounce else 0.0
                ),
                "avg_move_break": (
                    sum(move_break) / len(move_break) if move_break else 0.0
                ),
            }

    return stats


# ---------------------------
# NEAREST KEY LEVELS
# ---------------------------
def nearest_key_levels(
    levels_by_tf: Dict[str, Dict[float, Dict[str, int]]],
    current_price: float,
    num_levels: int = 3,
) -> Dict[str, List[Tuple[float, Dict[str, int]]]]:
    """Return the nearest key levels above and below the current price."""
    combined: Dict[float, Dict[str, int]] = {}
    for tf_levels in levels_by_tf.values():
        for price, data in tf_levels.items():
            if price not in combined:
                combined[price] = {"count": 0, "role": data["role"]}
            combined[price]["count"] += data["count"]

    all_levels = sorted(combined.items())
    below = [(p, d) for p, d in all_levels if p < current_price]
    above = [(p, d) for p, d in all_levels if p > current_price]

    return {
        "below": below[-num_levels:],
        "above": above[:num_levels],
    }


def print_recommendations(
    ticker: str, current_price: float, nearest: Dict[str, List[Tuple[float, Dict[str, int]]]]
) -> None:
    """Pretty-print nearest key levels for a ticker."""
    print(f"\nTicker: {ticker}  (close: {current_price:.2f})")
    print("  Below:")
    for price, data in reversed(nearest["below"]):
        print(f"    {price:.2f}  ({data['role']}, hits: {data['count']})")
    print("  Above:")
    for price, data in nearest["above"]:
        print(f"    {price:.2f}  ({data['role']}, hits: {data['count']})")


# ---------------------------
# MAIN ANALYSIS ROUTINES
# ---------------------------
def analyze_ticker(
    ticker: str,
    start: str,
    end: str | None,
    intervals: List[str],
) -> Tuple[
    float,
    Dict[str, List[Tuple[float, Dict[str, int]]]],
    Dict[str, Dict[float, Dict[str, int]]],
]:
    """Compute key levels for *ticker* and return nearest recommendations."""
    all_swings: List[pd.DataFrame] = []
    levels_by_tf: Dict[str, Dict[float, Dict[str, int]]] = {}

    for interval in intervals:
        df = download_data(ticker, start=start, end=end, interval=interval)
        if df.empty:
            continue
        swings = find_swing_points(df)
        if swings.empty:
            continue
        levels_by_tf[interval] = get_key_levels(swings)
        swings["Timeframe"] = interval
        all_swings.append(swings)

    if not all_swings:
        raise ValueError(f"No swing data found for {ticker}.")

    df_daily = download_data(ticker, start=start, end=end, interval="1d")
    latest_close = df_daily["Close"].iloc[-1]
    nearest = nearest_key_levels(levels_by_tf, latest_close)
    return latest_close, nearest, levels_by_tf


def run_analysis(tickers: List[str], start: str, intervals: List[str]) -> None:
    """Run analysis for a list of tickers and print recommendations."""
    for ticker in tickers:
        try:
            current_price, nearest, _ = analyze_ticker(ticker, start, None, intervals)
        except ValueError as exc:  # No data found
            print(f"{ticker}: {exc}")
            continue
        print_recommendations(ticker, current_price, nearest)


# ---------------------------
# BACKTESTING
# ---------------------------
def backtest_key_levels(
    df: pd.DataFrame, levels_by_tf: Dict[str, Dict[float, Dict[str, int]]], lookahead: int
) -> Dict[str, float]:
    """Simple backtest: count touches of key levels within a lookahead window."""
    hits = 0
    tests = 0

    for levels in levels_by_tf.values():
        for level in levels:
            for i in range(len(df) - lookahead):
                window = df.iloc[i + 1 : i + 1 + lookahead]
                if ((window["Low"] <= level) & (window["High"] >= level)).any():
                    hits += 1
                tests += 1

    hit_rate = hits / tests if tests else 0.0
    return {"hits": hits, "tests": tests, "hit_rate": hit_rate}


def run_backtest(
    tickers: List[str],
    train_start: str,
    train_end: str,
    test_end: str,
    intervals: List[str],
    lookahead: int,
) -> None:
    """Run backtests for tickers and print summary statistics."""
    for ticker in tickers:
        try:
            _, _, levels_by_tf = analyze_ticker(ticker, train_start, train_end, intervals)
        except ValueError as exc:
            print(f"{ticker}: {exc}")
            continue

        df = download_data(ticker, start=train_end, end=test_end, interval="1d")
        stats = backtest_key_levels(df, levels_by_tf, lookahead)
        print(
            f"{ticker}: hit rate {stats['hit_rate']:.2%} "
            f"({stats['hits']}/{stats['tests']})"
        )


# ---------------------------
# CLI & SCHEDULING
# ---------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-timeframe key level analyzer")
    parser.add_argument("--tickers", nargs="+", help="Tickers to monitor", required=True)
    parser.add_argument("--start", default="2024-01-01", help="Start date for historical data")
    parser.add_argument("--train-start", default=None, help="Training period start date (defaults to --start)")
    parser.add_argument("--train-end", default=None, help="Training period end date")
    parser.add_argument("--test-end", default=None, help="Test period end date")
    parser.add_argument("--intervals", nargs="*", default=["1d", "1h", "4h", "1wk"], help="Timeframes to analyze (e.g. 1d 1h)")
    parser.add_argument("--schedule", help="Optional HH:MM time to run each day; keeps the script alive", default=None)
    parser.add_argument("--backtest", type=int, default=None, help="Lookahead periods for key level backtesting")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    def job() -> None:
        if args.backtest is not None:
            if args.train_end is None or args.test_end is None:
                raise ValueError("--train-end and --test-end are required for backtesting")
            train_start = args.train_start or args.start
            run_backtest(
                args.tickers,
                train_start,
                args.train_end,
                args.test_end,
                args.intervals,
                args.backtest,
            )
        else:
            run_analysis(args.tickers, args.start, args.intervals)

    if args.schedule:
        if schedule is None:
            raise RuntimeError("schedule package is required for --schedule")
        schedule.every().day.at(args.schedule).do(job)
        print(f"Scheduled daily run at {args.schedule}. Press Ctrl+C to exit.")
        while True:
            schedule.run_pending()
            time.sleep(1)
    else:
        job()


if __name__ == "__main__":
    main()
