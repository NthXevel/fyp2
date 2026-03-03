"""
models/predictor.py — Loads trained XGBoost model and generates live trade signals.

Usage:
    from models.predictor import Predictor
    predictor = Predictor()
    result    = predictor.predict_latest(df)
"""
from __future__ import annotations

import pandas as pd
from models.trainer import ModelTrainer
from strategies.feature_engineering import FeatureEngineer
from config.settings import MODEL_PATH, CONFIDENCE_THRESHOLD


class Predictor:
    """
    Wraps a saved ModelTrainer and FeatureEngineer to produce
    trade signals (1 = long, -1 = short, 0 = flat) from raw OHLCV bars.
    """

    def __init__(self, model_path: str = MODEL_PATH,
                 threshold: float = CONFIDENCE_THRESHOLD):
        self.fe = FeatureEngineer()
        self.threshold = threshold

        # Load the saved model via ModelTrainer
        self.trainer = ModelTrainer(model_path=model_path)
        if not self.trainer.load():
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Train first with: python -m scripts.train"
            )
        self.feature_cols = self.trainer.feature_cols
        print(f"Predictor ready | threshold: {self.threshold}")

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals for a DataFrame of recent OHLCV bars.

        Returns a copy of *df* with added columns:
            prob_up  – probability that the next bar closes higher
            signal   – 1 (long), -1 (short), or 0 (flat)
        """
        df = self.fe.create_features(df)

        # Check for missing feature columns
        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            print(f"WARNING Missing features: {missing}")
            return df

        X = df[self.feature_cols].dropna()
        if X.empty:
            print("WARNING No valid rows after feature computation")
            return df

        df = df.loc[X.index].copy()
        df["prob_up"] = self.trainer.predict_proba(X)

        df["signal"] = 0
        df.loc[df["prob_up"] >= self.threshold, "signal"] = 1           # Long
        df.loc[df["prob_up"] <= (1 - self.threshold), "signal"] = -1    # Short

        n_long  = (df["signal"] == 1).sum()
        n_short = (df["signal"] == -1).sum()
        n_flat  = len(df) - n_long - n_short
        print(f"Signals: {n_long} long, {n_short} short, {n_flat} flat")

        return df

    def predict_latest(self, df: pd.DataFrame) -> dict:
        """
        Convenience method — return the signal for the most recent bar only.

        Returns:
            dict with keys: datetime, close, prob_up, signal
        """
        result_df = self.predict(df)
        latest = result_df.iloc[-1]
        return {
            "datetime": str(latest.name),
            "close":    float(latest.get("close", 0)),
            "prob_up":  float(latest["prob_up"]),
            "signal":   int(latest["signal"]),
        }
