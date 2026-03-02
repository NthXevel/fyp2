"""
Data Downloader — Downloads historical stock data to local CSV files.
Uses Yahoo Finance as data source.

Usage:
    python -m scripts.download_data                                    # Download TRAINING_SYMBOLS, daily, 2yr
    python -m scripts.download_data --symbol TSLA                      # Single symbol
    python -m scripts.download_data --symbols AAPL TSLA MSFT           # Multiple symbols
    python -m scripts.download_data --days 730 --interval 1d           # 2 years daily
"""

import argparse
import os
import sys
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf
from config.settings import STOCK_SYMBOL, TRAINING_SYMBOLS, TRAINING_INTERVAL, TRAINING_DAYS

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


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
                       help="Single ticker symbol to download")
    group.add_argument("--symbols", nargs="+", type=str, default=None,
                       help="Multiple ticker symbols to download")
    parser.add_argument("--days", type=int, default=TRAINING_DAYS,
                        help=f"Days of history (default: {TRAINING_DAYS})")
    parser.add_argument("--interval", type=str, default=TRAINING_INTERVAL,
                        choices=["1m", "2m", "5m", "15m", "30m", "60m", "90m",
                                 "1h", "1d", "5d", "1wk", "1mo", "3mo"],
                        help=f"Data interval (default: {TRAINING_INTERVAL})")
    args = parser.parse_args()

    # Resolve symbol list
    if args.symbols:
        symbols = [s.upper() for s in args.symbols]
    elif args.symbol:
        symbols = [args.symbol.upper()]
    else:
        symbols = [s.upper() for s in TRAINING_SYMBOLS]

    print(f"\n{'=' * 50}")
    print(f"Stock Data Downloader  (YAHOO)")
    print(f"Symbols : {', '.join(symbols)}")
    print(f"Days    : {args.days}")
    print(f"Interval: {args.interval}")
    print(f"{'=' * 50}\n")

    saved_files = []
    for symbol in symbols:
        df = download_stock_data(symbol, days=args.days, interval=args.interval)

        if df is not None:
            path = save_to_csv(df, symbol, args.interval)
            saved_files.append(path)
        print()

    # Summary
    print(f"{'=' * 50}")
    if saved_files:
        print(f"Done! {len(saved_files)} file(s) saved to '{DATA_DIR}':")
        for f in saved_files:
            print(f"  - {f}")
    else:
        print("No data was downloaded.")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
