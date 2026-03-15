"""
Accumulate intraday bars — run periodically (e.g. daily via Task Scheduler)
to upsert fresh 15-minute bars into PostgreSQL.

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
from config.settings import TRAINING_SYMBOLS, yahoo_symbol
from utils.db_connector import init_database, upsert_market_data


def _yahoo_fetch(symbol: str, interval: str, days: int) -> pd.DataFrame | None:
    """Fetch recent bars from Yahoo Finance."""
    end = datetime.now()
    start = end - timedelta(days=days)
    try:
        df = yf.download(yahoo_symbol(symbol), start=start, end=end,
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
    Download recent bars and upsert them into PostgreSQL.
    """
    # Fetch new data
    new_df = _yahoo_fetch(symbol, interval, days)

    if new_df is None or new_df.empty:
        print(f"  [{symbol}] No new data fetched.")
        return

    rows = upsert_market_data(new_df, symbol=symbol, timeframe=interval, source="yahoo")
    print(f"  [{symbol}] upserted {rows} row(s) into market_data")


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

    init_database()

    for sym in symbols:
        accumulate(sym, args.interval, args.days)

    print(f"\n{'=' * 55}")
    print("  Done!")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    main()
