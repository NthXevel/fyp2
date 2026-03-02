"""
Main script for training the XGBoost model

Supports:
    • Single-stock training  (default STOCK_SYMBOL)
    • Multi-stock training   (TRAINING_SYMBOLS from .env)
    • Daily data (years of history via Yahoo)
    • Alpaca intraday data   (--source alpaca for long 15m history)

Usage:
    python -m scripts.train                          # Multi-stock, daily, Yahoo
    python -m scripts.train --single                 # Single-stock only
    python -m scripts.train --source alpaca           # Use Alpaca for intraday
    python -m scripts.train --interval 15m --days 730 # Custom interval/lookback
    python -m scripts.train --trials 100              # More Optuna trials

Targets:
    • Accuracy  ≥ 55 %
    • Sharpe    ≥ 0.5
    • Max DD    ≤ 10 %
"""
import argparse
import os
import sys
import pandas as pd

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.data_fetcher import DataFetcher
from strategies.feature_engineering import FeatureEngineer
from models.trainer import ModelTrainer
from config.settings import (
    STOCK_SYMBOL, DATA_INTERVAL, DATA_DAYS,
    TRAINING_SYMBOLS, TRAINING_INTERVAL, TRAINING_DAYS,
    TARGET_ACCURACY, TARGET_SHARPE, TARGET_MAX_DRAWDOWN,
    safe_filename,
)


# Map user-facing intervals to Alpaca timeframe strings
_ALPACA_TF = {
    '1m': '1Min', '5m': '5Min', '15m': '15Min',
    '30m': '30Min', '1h': '1Hour', '1d': '1Day',
}


def load_symbol_data(symbol: str, interval: str, days: int,
                     source: str) -> pd.DataFrame | None:
    """
    Load OHLCV data for a single symbol.  Tries local CSV first,
    then falls back to Yahoo or Alpaca.
    """
    csv_path = os.path.join('data', f'{safe_filename(symbol)}_{interval}.csv')

    if os.path.exists(csv_path):
        print(f"  Loading local CSV: {csv_path}")
        df = pd.read_csv(csv_path, index_col='date', parse_dates=True)
        return df

    if source == 'alpaca':
        tf = _ALPACA_TF.get(interval, interval)
        print(f"  Downloading {symbol} via Alpaca ({tf}, {days}d) ...")
        df = DataFetcher.get_alpaca_bars(symbol, interval=tf, days=days)
    else:
        print(f"  Downloading {symbol} via Yahoo ({interval}, {days}d) ...")
        fetcher = DataFetcher(symbol=symbol, interval=interval)
        df = fetcher.get_historical_data(days=days, interval=interval)

    return df


def main():
    parser = argparse.ArgumentParser(description="Train the XGBoost model.")
    parser.add_argument("--single", action="store_true",
                        help="Train on STOCK_SYMBOL only (no multi-stock)")
    parser.add_argument("--symbols", nargs="+", default=None,
                        help="Override TRAINING_SYMBOLS list")
    parser.add_argument("--interval", default=None,
                        help="Bar interval (default: TRAINING_INTERVAL from .env)")
    parser.add_argument("--days", type=int, default=None,
                        help="Days of history (default: TRAINING_DAYS from .env)")
    parser.add_argument("--source", default="yahoo",
                        choices=["yahoo", "alpaca"],
                        help="Data source (default: yahoo)")
    parser.add_argument("--trials", type=int, default=50,
                        help="Number of Optuna trials (default: 50)")
    args = parser.parse_args()

    interval = args.interval or TRAINING_INTERVAL
    days     = args.days     or TRAINING_DAYS

    if args.single:
        symbols = [STOCK_SYMBOL]
    elif args.symbols:
        symbols = [s.upper() for s in args.symbols]
    else:
        symbols = TRAINING_SYMBOLS

    mode = "single-stock" if len(symbols) == 1 else "multi-stock"

    print("=" * 60)
    print(f"Training XGBoost Model ({mode})")
    print(f"Symbols  : {', '.join(symbols)}")
    print(f"Interval : {interval}  |  History: {days} days")
    print(f"Source   : {args.source.upper()}")
    print(f"Optuna   : {args.trials} trials")
    print("=" * 60)

    # ── Step 1: Fetch data for each symbol ──────────────────────────
    print(f"\n[1/4] Fetching historical data for {len(symbols)} symbol(s) ...")
    dataframes: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = load_symbol_data(sym, interval, days, args.source)
        if df is not None and not df.empty:
            print(f"    {sym}: {len(df)} bars  "
                  f"({df.index[0]} → {df.index[-1]})")
            dataframes[sym] = df
        else:
            print(f"    {sym}: ⚠ no data — skipping")

    if not dataframes:
        print("No data fetched for any symbol. Exiting.")
        sys.exit(1)

    # ── Step 2: Engineer features ───────────────────────────────────
    print(f"\n[2/4] Engineering features ...")
    engineer = FeatureEngineer()

    if len(dataframes) == 1:
        sym = list(dataframes.keys())[0]
        df = engineer.create_features(dataframes[sym], symbol=sym)
        X, y, feature_cols = engineer.prepare_training_data(df)
    else:
        X, y, feature_cols = engineer.prepare_multi_stock(dataframes)

    print(f"  Features       : {len(feature_cols)}")
    print(f"  Training rows  : {len(X)}")
    print(f"  Target balance : {y.value_counts().to_dict()}")

    # ── Step 3: Train model ─────────────────────────────────────────
    print(f"\n[3/4] Training XGBoost model (Optuna, {args.trials} trials) ...")
    trainer = ModelTrainer()
    accuracy = trainer.train(X, y, feature_cols, n_tune_trials=args.trials)

    # ── Step 4: Save model ──────────────────────────────────────────
    print("\n[4/4] Saving model ...")
    trainer.save()

    # Feature importance
    print("\nTop 10 Important Features:")
    importance = trainer.get_feature_importance(top_n=10)
    for i, (feature, score) in enumerate(importance.items(), 1):
        print(f"  {i}. {feature}: {score:.4f}")

    print("\n" + "=" * 60)
    status = "✓ MET" if accuracy >= TARGET_ACCURACY else "✗ MISSED"
    print(f"Training complete!  Accuracy: {accuracy:.4f}  "
          f"({status} target ≥ {TARGET_ACCURACY})")
    print(f"Symbols used: {', '.join(dataframes.keys())}")
    print("=" * 60)


if __name__ == "__main__":
    main()
