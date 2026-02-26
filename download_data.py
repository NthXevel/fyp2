"""
Data Downloader - Downloads historical stock data to local CSV files.

Usage:
    python download_data.py                        # Download default symbol (from .env) with 1 year of data
    python download_data.py --symbol TSLA          # Download specific symbol
    python download_data.py --symbols AAPL TSLA MSFT  # Download multiple symbols
    python download_data.py --days 730             # Download 2 years of data
    python download_data.py --interval 1h          # Download hourly data (max 730 days)
"""

import argparse
import os
import sys
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf
from src.config import STOCK_SYMBOL

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def download_stock_data(symbol: str, days: int = 365, interval: str = "1d") -> pd.DataFrame | None:
    """
    Download historical OHLCV data for a given symbol from Yahoo Finance.

    Args:
        symbol:   Ticker symbol (e.g. 'AAPL').
        days:     Number of calendar days of history to fetch.
        interval: Data interval – '1d', '1h', '5m', etc.

    Returns:
        A cleaned DataFrame, or None on failure.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    print(f"Downloading {symbol} | interval={interval} | "
          f"{start_date.strftime('%Y-%m-%d')} -> {end_date.strftime('%Y-%m-%d')} ...")

    try:
        df = yf.download(
            symbol,
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


def save_to_csv(df: pd.DataFrame, symbol: str, interval: str) -> str:
    """Save a DataFrame to data/<SYMBOL>_<interval>.csv and return the path."""
    os.makedirs(DATA_DIR, exist_ok=True)
    filename = f"{symbol.upper()}_{interval}.csv"
    filepath = os.path.join(DATA_DIR, filename)
    df.to_csv(filepath)
    print(f"  -> Saved to {filepath}")
    return filepath


def main():
    parser = argparse.ArgumentParser(description="Download historical stock data to CSV.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--symbol", type=str, default=None,
                       help="Single ticker symbol to download (default: from .env)")
    group.add_argument("--symbols", nargs="+", type=str, default=None,
                       help="Multiple ticker symbols to download")
    parser.add_argument("--days", type=int, default=59,
                        help="Number of calendar days of history (default: 59, max ~60 for 15m)")
    parser.add_argument("--interval", type=str, default="15m",
                        choices=["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"],
                        help="Data interval (default: 15m)")
    args = parser.parse_args()

    # Resolve symbol list
    if args.symbols:
        symbols = [s.upper() for s in args.symbols]
    elif args.symbol:
        symbols = [args.symbol.upper()]
    else:
        symbols = [STOCK_SYMBOL.upper()]

    print(f"\n{'='*50}")
    print(f"Stock Data Downloader")
    print(f"Symbols : {', '.join(symbols)}")
    print(f"Days    : {args.days}")
    print(f"Interval: {args.interval}")
    print(f"{'='*50}\n")

    saved_files = []
    for symbol in symbols:
        df = download_stock_data(symbol, days=args.days, interval=args.interval)
        if df is not None:
            path = save_to_csv(df, symbol, args.interval)
            saved_files.append(path)
        print()

    # Summary
    print(f"{'='*50}")
    if saved_files:
        print(f"Done! {len(saved_files)} file(s) saved to '{DATA_DIR}':")
        for f in saved_files:
            print(f"  - {f}")
    else:
        print("No data was downloaded.")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
