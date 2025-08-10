# Forex SMC Toolkit

Downloads OHLC data for forex pairs (15m, 4h, 1d, 1w) and computes:
- Fair Value Gaps (FVG)
- Break of Structure (BOS) and Change of Character (CHOCH)
- Premium / Equilibrium / Discount zones (4h, 1d, 1w)

Data providers: Yahoo Finance by default, with automatic Alpha Vantage fallback. To enable fallback, set environment variable `ALPHAVANTAGE_API_KEY`.

## Setup

```bash
pip install -r requirements.txt
# If pip is system-managed (PEP 668), use:
python3 -m pip install --user --break-system-packages -r requirements.txt
```

## Usage

```bash
# Optional: enables fallback provider if Yahoo fails
export ALPHAVANTAGE_API_KEY=YOUR_KEY

python main.py --symbols EURUSD GBPUSD USDJPY --out ./output
```

Options:
- `--symbols` Space-separated list of pairs (format: EURUSD). Converts automatically.
- `--max-workers` Max concurrent symbols (default: 4).
- `--out` Output directory (default: `./output`).

Outputs:
- `output/data/{PAIR}/{tf}.csv` Enriched OHLC per timeframe (`15m`, `4h`, `1d`, `1w`).
- `output/reports/{PAIR}.json` Summary of latest BOS/CHOCH, FVGs, and zone levels.