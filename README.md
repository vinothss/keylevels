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

To evaluate how price has historically reacted to the detected levels, use the
`--backtest` flag. The numeric argument sets the *lookahead* window in bars for
checking whether price bounces or breaks a level. Specify the training and
testing periods with `--train-start`, `--train-end` and `--test-end`:

```bash
python keylevels.py --tickers AAPL --backtest 10 --train-start 2023-01-01 --train-end 2023-12-31 --test-end 2024-06-01
```

The backtest currently reports:

* **Bounce ratio** – percentage of touches that result in a bounce within the
  lookahead window.
* **Average move** – mean price change following bounces or breaks.

Levels are detected using the training period and behaviour is evaluated on
data after `--train-end` up to `--test-end`, inspecting the next `N` bars where
`N` is the `--backtest` value.

No extra dependencies are required beyond those in `requirements.txt`, but make
sure recent versions of `pandas` and `yfinance` are installed (e.g.
`pandas>=1.5` and `yfinance>=0.2`) so the backtest calculations work properly.
