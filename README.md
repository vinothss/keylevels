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
from `2024-01-01` (configurable via `--train-start`) and prints the nearest key
levels above and below the latest close. Dates should be supplied in
`YYYY-MM-DD` format.

To keep the script running and execute the analysis every day at a specific
local time use the `--schedule` option (requires the `schedule` package):

```bash
python keylevels.py --tickers AAPL MSFT --schedule 16:00
```

The tool can be scheduled by external systems such as cron as well.

## Backtesting


To evaluate how price has historically reacted to the detected levels, use the
`--backtest` flag along with separate training and testing windows. `--train-start`
defines the beginning of the training period used for level detection, while
`--test-start` marks the beginning of the testing period. The numeric argument
to `--backtest` sets the *lookahead* window in bars for checking whether price
bounces or breaks a level.

```bash
python keylevels.py --tickers AAPL --backtest 10 --train-start 2020-01-01 --test-start 2024-01-01
```

The training window must start before the testing window; otherwise the program
will raise an error.

The backtest currently reports:

```text
AAPL: hit rate 21.74% (10/46)
```

During backtesting, levels are detected using data from the training window
and then evaluated on the separate testing window. Each touch is examined over
the next `N` bars, where `N` is the value supplied to `--backtest`.

No extra dependencies are required beyond those in `requirements.txt`, but make
sure recent versions of `pandas` and `yfinance` are installed (e.g.
`pandas>=1.5` and `yfinance>=0.2`) so the backtest calculations work properly.
