# keylevels

A small utility for computing multi-timeframe support and resistance levels for
stocks using data from Yahoo Finance.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the script once for a list of tickers:

```bash
python keylevels.py --tickers AAPL MSFT
```

By default the script analyses daily, hourly, 4‑hour and weekly data starting
from `2024-01-01` and prints the nearest key levels above and below the latest
close.

To keep the script running and execute the analysis every day at a specific
local time use the `--schedule` option (requires the `schedule` package):

```bash
python keylevels.py --tickers AAPL MSFT --schedule 16:00
```

The tool can be scheduled by external systems such as cron as well.

## Backtesting

To estimate how often price revisits detected levels, use the `--backtest`
flag. The numeric argument sets the *lookahead* window in bars for counting
future touches of each level.

```bash
python keylevels.py --tickers AAPL --backtest 10
```

The backtest summarises how many potential tests occurred and how many were
actual hits, reporting a hit rate:

```text
AAPL: hit rate 21.74% (10/46)
```

The same historical period is used to derive the levels and to evaluate them,
so the split is effectively in-sample. Levels remain static during testing and
no out-of-sample validation is performed, which can lead to optimistic hit
rates.

No extra dependencies are required beyond those in `requirements.txt`, but make
sure recent versions of `pandas` and `yfinance` are installed (e.g.
`pandas>=1.5` and `yfinance>=0.2`) so the backtest calculations work properly.
