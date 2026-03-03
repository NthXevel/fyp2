"""
Accumulate intraday bars — run periodically (e.g. daily via cron / Task Scheduler)
to grow your local CSV with new 15-minute bars.

Usage:
    python -m scripts.accumulate_data                          # Append for default symbol
    python -m scripts.accumulate_data --symbols BTC/USD ETH/USD  # Multiple symbols
    python -m scripts.accumulate_data --interval 1h --days 5    # Custom interval
"""
import argparse
import os
import sys
import pandas as pd
from datetime import datetime, timedelta

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import yfinance as yf
from config.settings import STOCK_SYMBOL, TRAINING_SYMBOLS

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def _yahoo_fetch(symbol: str, interval: str, days: int) -> pd.DataFrame | None:
    """Fetch recent bars from Yahoo Finance."""
    end = datetime.now()
    start = end - timedelta(days=days)
    try:
        df = yf.download(symbol, start=start, end=end,
                         interval=interval, progress=False)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.rename(columns={
            'Open': 'open', 'High': 'high', 'Low': 'low',
            'Close': 'close', 'Volume': 'volume',
        })
        df = df[['open', 'high', 'low', 'close', 'volume']]
        df.index.name = 'date'
        return df
    except Exception as e:
        print(f"  [Yahoo] Error for {symbol}: {e}")
        return None


def accumulate(symbol: str, interval: str, days: int):
    """
    Download recent bars and append only the NEW rows to the local CSV.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    csv_path = os.path.join(DATA_DIR, f"{symbol}_{interval}.csv")

    # Fetch new data
    new_df = _yahoo_fetch(symbol, interval, days)

    if new_df is None or new_df.empty:
        print(f"  [{symbol}] No new data fetched.")
        return

    # Load existing CSV (if any) and merge
    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path, index_col='date', parse_dates=True)
        before = len(existing)
        combined = pd.concat([existing, new_df])
        combined = combined[~combined.index.duplicated(keep='last')]
        combined.sort_index(inplace=True)
        new_rows = len(combined) - before
    else:
        combined = new_df
        new_rows = len(combined)

    combined.to_csv(csv_path)
    print(f"  [{symbol}] {new_rows} new rows appended -> {csv_path}  "
          f"(total: {len(combined)} rows)")


def main():
    parser = argparse.ArgumentParser(
        description="Append new bars to local CSVs (run periodically).")
    parser.add_argument("--symbols", nargs="+", default=None,
                        help="Symbols to accumulate (default: TRAINING_SYMBOLS)")
    parser.add_argument("--interval", default="15m",
                        help="Bar interval (default: 15m)")
    parser.add_argument("--days", type=int, default=7,
                        help="Days of recent data to fetch (default: 7)")
    args = parser.parse_args()

    symbols = [s.upper() for s in (args.symbols or TRAINING_SYMBOLS)]

    print(f"\n{'=' * 55}")
    print(f"  Data Accumulator — YAHOO")
    print(f"  Symbols : {', '.join(symbols)}")
    print(f"  Interval: {args.interval}  |  Last {args.days} days")
    print(f"{'=' * 55}\n")

    for sym in symbols:
        accumulate(sym, args.interval, args.days)

    print(f"\n{'=' * 55}")
    print("  Done!")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    main()
