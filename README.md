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
