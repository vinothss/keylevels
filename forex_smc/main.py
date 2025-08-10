from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from smc.data import get_timeframes_data
from smc.fvg import detect_fvg, latest_fvgs
from smc.structure import annotate_structure
from smc.zones import compute_zones, summarize_zone_levels
from smc.utils import to_yahoo_symbol


TF_ORDER = ["15m", "4h", "1d", "1w"]


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    df1 = detect_fvg(df)
    df2 = annotate_structure(df1)
    df3 = compute_zones(df2)
    return df3


def process_symbol(symbol: str, out_dir: Path) -> Dict:
    data = get_timeframes_data(symbol)

    pair = symbol.upper().replace("=X", "")
    pair_dir = out_dir / "data" / pair
    pair_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "symbol": pair,
        "timeframes": {},
    }

    for tf in TF_ORDER:
        df = data[tf]
        if df.empty:
            continue
        enriched = enrich(df)
        enriched.to_csv(pair_dir / f"{tf}.csv")

        fvgs = latest_fvgs(enriched)
        zones = summarize_zone_levels(enriched)
        last_row = enriched.iloc[-1]
        last_events = {
            "bos_up": bool(last_row.get("bos_up", False)),
            "bos_down": bool(last_row.get("bos_down", False)),
            "choch_up": bool(last_row.get("choch_up", False)),
            "choch_down": bool(last_row.get("choch_down", False)),
            "structure_trend": last_row.get("structure_trend", None),
        }

        report["timeframes"][tf] = {
            "fvgs": fvgs,
            "zones": zones,
            "last_events": last_events,
        }

    reports_dir = out_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / f"{pair}.json", "w") as f:
        json.dump(report, f, indent=2)

    return report


def main():
    parser = argparse.ArgumentParser(description="Forex SMC downloader and analyzer")
    parser.add_argument("--symbols", nargs="+", required=True, help="Forex pairs like EURUSD GBPUSD USDJPY")
    parser.add_argument("--out", default="./output", help="Output directory")
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()

    symbols = [to_yahoo_symbol(s) for s in args.symbols]
    out_dir = Path(args.out)

    results: List[Dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futs = {ex.submit(process_symbol, s, out_dir): s for s in symbols}
        for fut in concurrent.futures.as_completed(futs):
            try:
                results.append(fut.result())
            except Exception as e:
                sym = futs[fut]
                print(f"Error processing {sym}: {e}")

    # Write an index file
    index = {r["symbol"]: r["timeframes"] for r in results}
    (out_dir / "reports").mkdir(parents=True, exist_ok=True)
    with open(out_dir / "reports" / "index.json", "w") as f:
        json.dump(index, f, indent=2)


if __name__ == "__main__":
    main()