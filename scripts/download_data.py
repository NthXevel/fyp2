"""
Data Downloader — Downloads historical stock data and upserts into PostgreSQL.
Uses Yahoo Finance as data source.

Usage:
    python -m scripts.download_data                                    # Download TRAINING_SYMBOLS, daily, 2yr
    python -m scripts.download_data --symbol TSLA                      # Single symbol
    python -m scripts.download_data --symbols BTC/USD ETH/USD     # Multiple symbols
    python -m scripts.download_data --days 730 --interval 1d           # 2 years daily
"""

import argparse
import os
import sys
from datetime import datetime, timedelta

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import yfinance as yf
from config.settings import (
    STOCK_SYMBOL, TRAINING_SYMBOLS, TRAINING_INTERVAL, TRAINING_DAYS,
    TRAINING_INTERVALS, TRAINING_DAYS_15M,
    yahoo_symbol,
)
from utils.db_connector import init_database, upsert_market_data


def download_stock_data(symbol: str, days: int = 365, interval: str = "1d") -> pd.DataFrame | None:
    """
    Download historical OHLCV data for a given symbol from Yahoo Finance.

    Args:
        symbol:   Ticker symbol ('BTC/USD').
        days:     Number of calendar days of history to fetch.
        interval: Data interval – '1d', '1h', '5m', etc.

    Returns:
        A cleaned DataFrame, or None on failure.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    yf_sym = yahoo_symbol(symbol)  # BTC/USD -> BTC-USD for Yahoo
    print(f"Downloading {symbol} ({yf_sym}) | interval={interval} | "
          f"{start_date.strftime('%Y-%m-%d')} -> {end_date.strftime('%Y-%m-%d')} ...")

    try:
        df = yf.download(
            yf_sym,
            start=start_date,
            end=end_date,
            interval=interval,
            progress=False,
        )

        if df is None or df.empty:
            print(f"  [!] No data returned for {symbol}.")
            return None

        # Flatten MultiIndex columns (single-ticker download quirk)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Normalise column names to lowercase
        df = df.rename(columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        })

        # Keep only the columns we care about
        keep = [c for c in ["open", "high", "low", "close", "adj_close", "volume"] if c in df.columns]
        df = df[keep]

        df.index.name = "date"
        print(f"  [OK] {len(df)} rows fetched for {symbol}.")
        return df

    except Exception as exc:
        print(f"  [ERROR] Failed to download {symbol}: {exc}")
        return None


def save_to_db(df: pd.DataFrame, symbol: str, interval: str) -> int:
    """Upsert DataFrame rows into market_data table and return row count."""
    rows = upsert_market_data(df, symbol=symbol, timeframe=interval, source="yahoo")
    print(f"  -> Upserted {rows} row(s) into market_data")
    return rows


def main():
    parser = argparse.ArgumentParser(description="Download historical stock data to PostgreSQL.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--symbol", type=str, default=None,
                       help="Single ticker symbol to download")
    group.add_argument("--symbols", nargs="+", type=str, default=None,
                       help="Multiple ticker symbols to download")
    parser.add_argument("--days", type=int, default=None,
                        help=f"Days of history (default: auto per interval)")
    parser.add_argument("--interval", type=str, default=None,
                        choices=["1m", "2m", "5m", "15m", "30m", "60m", "90m",
                                 "1h", "1d", "5d", "1wk", "1mo", "3mo"],
                        help="Data interval (default: download all TRAINING_INTERVALS)")
    args = parser.parse_args()

    # Resolve symbol list
    if args.symbols:
        symbols = [s.upper() for s in args.symbols]
    elif args.symbol:
        symbols = [args.symbol.upper()]
    else:
        symbols = [s.upper() for s in TRAINING_SYMBOLS]

    # Resolve intervals to download
    if args.interval:
        intervals = [args.interval]
    else:
        intervals = TRAINING_INTERVALS          # e.g. ['1d', '15m']

    print(f"\n{'=' * 50}")
    print(f"Stock Data Downloader  (YAHOO)")
    print(f"Symbols   : {', '.join(symbols)}")
    print(f"Intervals : {', '.join(intervals)}")
    print(f"{'=' * 50}\n")

    init_database()
    total_rows = 0
    for interval in intervals:
        # Choose appropriate lookback per interval
        if args.days:
            days = args.days
        elif interval == '15m':
            days = TRAINING_DAYS_15M
        else:
            days = TRAINING_DAYS

        print(f"--- interval={interval}  days={days} ---")
        for symbol in symbols:
            df = download_stock_data(symbol, days=days, interval=interval)

            if df is not None:
                total_rows += save_to_db(df, symbol, interval)
            print()

    # Summary
    print(f"{'=' * 50}")
    if total_rows > 0:
        print(f"Done! {total_rows} row(s) written to PostgreSQL market_data.")
    else:
        print("No data was downloaded or persisted.")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
