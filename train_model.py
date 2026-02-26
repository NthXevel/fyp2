"""
Main script for training the XGBoost model
Run this script to fetch data and train the model

Targets:
    • Accuracy  ≥ 55 %
    • Sharpe    ≥ 0.5
    • Max DD    ≤ 10 %
"""
import os
import sys
import pandas as pd
from src.data_fetcher import DataFetcher
from src.feature_engineering import FeatureEngineer
from src.model_trainer import ModelTrainer
from src.config import (
    STOCK_SYMBOL, DATA_INTERVAL, DATA_DAYS,
    TARGET_ACCURACY, TARGET_SHARPE, TARGET_MAX_DRAWDOWN,
)


def main():
    print("=" * 60)
    print(f"Training XGBoost Model for {STOCK_SYMBOL} ({DATA_INTERVAL} interval)")
    print("=" * 60)
    
    # Step 1: Fetch historical data (try local CSV first)
    print("\n[1/4] Fetching historical data...")
    csv_path = os.path.join('data', f'{STOCK_SYMBOL}_{DATA_INTERVAL}.csv')
    
    if os.path.exists(csv_path):
        print(f"Loading local CSV: {csv_path}")
        df = pd.read_csv(csv_path, index_col='date', parse_dates=True)
    else:
        print(f"Downloading {STOCK_SYMBOL} {DATA_INTERVAL} data from Yahoo Finance...")
        fetcher = DataFetcher()
        df = fetcher.get_historical_data(days=DATA_DAYS)
    
    if df is None:
        print("Failed to fetch data. Exiting.")
        sys.exit(1)
    
    print(f"Fetched {len(df)} bars of {DATA_INTERVAL} data")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    # Step 2: Engineer features
    print("\n[2/4] Engineering features...")
    engineer = FeatureEngineer()
    df = engineer.create_features(df)
    X, y, feature_cols = engineer.prepare_training_data(df)
    
    print(f"Created {len(feature_cols)} features")
    print(f"Training samples: {len(X)}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    # Step 3: Train model
    print("\n[3/4] Training XGBoost model...")
    trainer = ModelTrainer()
    accuracy = trainer.train(X, y, feature_cols)
    
    # Step 4: Save model
    print("\n[4/4] Saving model...")
    trainer.save()
    
    # Display feature importance
    print("\nTop 10 Important Features:")
    importance = trainer.get_feature_importance(top_n=10)
    for i, (feature, score) in enumerate(importance.items(), 1):
        print(f"  {i}. {feature}: {score}")
    
    print("\n" + "=" * 60)
    status = "✓ MET" if accuracy >= TARGET_ACCURACY else "✗ MISSED"
    print(f"Training complete!  Accuracy: {accuracy:.4f}  ({status} target ≥ {TARGET_ACCURACY})")
    print("=" * 60)


if __name__ == "__main__":
    main()
