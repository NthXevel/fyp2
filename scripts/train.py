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
    TRAINING_INTERVALS, TRAINING_DAYS_15M,
    TARGET_ACCURACY, TARGET_SHARPE, TARGET_MAX_DRAWDOWN,
    TEST_SIZE, safe_filename,
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
    parser.add_argument("--source", default="yahoo",
                        choices=["yahoo", "alpaca"],
                        help="Data source (default: yahoo)")
    parser.add_argument("--trials", type=int, default=50,
                        help="Number of Optuna trials (default: 50)")
    parser.add_argument("--test-ratio", type=float, default=TEST_SIZE,
                        help="Fraction of BTC/USD 15m held for test (default: 0.2)")
    args = parser.parse_args()

    interval = '15m'                          # 15-minute bars only
    days     = TRAINING_DAYS_15M              # ~59 days (Yahoo max)

    if args.single:
        symbols = [STOCK_SYMBOL]
    elif args.symbols:
        symbols = [s.upper() for s in args.symbols]
    else:
        symbols = TRAINING_SYMBOLS

    # The dataset whose tail is held out for testing
    TEST_SYMBOL = 'BTC/USD'

    mode = "single-stock" if len(symbols) == 1 else "multi-stock"

    print("=" * 60)
    print(f"Training XGBoost Model ({mode}, 15m only)")
    print(f"Symbols   : {', '.join(symbols)}")
    print(f"Interval  : {interval}  |  History: {days} days")
    print(f"Source    : {args.source.upper()}")
    print(f"Optuna    : {args.trials} trials")
    print(f"Test set  : last {args.test_ratio:.0%} of {TEST_SYMBOL} 15m")
    print("=" * 60)

    # ── Step 1: Fetch 15m data for every symbol ─────────────────────
    print(f"\n[1/5] Fetching 15m data for {len(symbols)} symbol(s) ...")
    dataframes: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = load_symbol_data(sym, interval, days, args.source)
        if df is not None and not df.empty:
            print(f"    {sym}: {len(df)} bars  "
                  f"({df.index[0]} -> {df.index[-1]})")
            dataframes[sym] = df
        else:
            print(f"    {sym}: WARNING no data – skipping")

    if not dataframes:
        print("No data fetched for any symbol. Exiting.")
        sys.exit(1)

    # ── Step 2: Engineer features per symbol ────────────────────────
    print(f"\n[2/5] Engineering features ...")
    engineer = FeatureEngineer()
    featured: dict[str, pd.DataFrame] = {}
    for sym, df in dataframes.items():
        feat_df = engineer.create_features(df, symbol=sym)
        feat_df = feat_df.dropna(subset=engineer.feature_cols + ['Target'])
        featured[sym] = feat_df
        print(f"    {sym}: {len(feat_df)} usable rows")

    # ── Step 3: Split — hold out last 20% of BTC/USD for test ──────
    print(f"\n[3/5] Splitting data ...")
    feature_cols = engineer.feature_cols
    train_frames = []

    if TEST_SYMBOL in featured:
        test_df   = featured[TEST_SYMBOL]
        split_idx = int(len(test_df) * (1 - args.test_ratio))
        train_part = test_df.iloc[:split_idx]
        holdout    = test_df.iloc[split_idx:]
        train_frames.append(train_part)
        print(f"  {TEST_SYMBOL}: {len(train_part)} rows -> train, "
              f"{len(holdout)} rows -> test")
    else:
        holdout = None
        print(f"  WARNING: {TEST_SYMBOL} not found — no dedicated test set")

    # Everything else goes entirely into training
    for sym, df in featured.items():
        if sym != TEST_SYMBOL:
            train_frames.append(df)
            print(f"  {sym}: {len(df)} rows -> train (100%)")

    # Normalise tz-aware / tz-naive indices before concat
    for i, f in enumerate(train_frames):
        if getattr(f.index, 'tz', None) is not None:
            train_frames[i] = f.copy()
            train_frames[i].index = f.index.tz_localize(None)
    if holdout is not None and getattr(holdout.index, 'tz', None) is not None:
        holdout = holdout.copy()
        holdout.index = holdout.index.tz_localize(None)

    combined_train = pd.concat(train_frames).sort_index()
    X_train = combined_train[feature_cols]
    y_train = combined_train['Target']

    if holdout is not None:
        X_test = holdout[feature_cols]
        y_test = holdout['Target']
    else:
        # Fallback: chronological split on combined data
        n = int(len(X_train) * (1 - args.test_ratio))
        X_test  = X_train.iloc[n:]
        y_test  = y_train.iloc[n:]
        X_train = X_train.iloc[:n]
        y_train = y_train.iloc[:n]

    print(f"\n  Total training rows : {len(X_train)}")
    print(f"  Total test rows     : {len(X_test)}")
    print(f"  Features            : {len(feature_cols)}")
    print(f"  Train target balance: {y_train.value_counts().to_dict()}")
    print(f"  Test  target balance: {y_test.value_counts().to_dict()}")

    # ── Step 4: Train model ─────────────────────────────────────────
    print(f"\n[4/5] Training XGBoost model (Optuna, {args.trials} trials) ...")
    trainer = ModelTrainer()
    accuracy = trainer.train_with_split(
        X_train, y_train, X_test, y_test,
        feature_cols, n_tune_trials=args.trials,
    )

    # ── Step 5: Save model ──────────────────────────────────────────
    print("\n[5/5] Saving model ...")
    trainer.save()

    # Feature importance
    print("\nTop 10 Important Features:")
    importance = trainer.get_feature_importance(top_n=10)
    for i, (feature, score) in enumerate(importance.items(), 1):
        print(f"  {i}. {feature}: {score:.4f}")

    print("\n" + "=" * 60)
    status = "MET" if accuracy >= TARGET_ACCURACY else "MISSED"
    print(f"Training complete!  Accuracy: {accuracy:.4f}  "
          f"({status} target >= {TARGET_ACCURACY})")
    print(f"Symbols used: {', '.join(dataframes.keys())}")
    print("=" * 60)


if __name__ == "__main__":
    main()
