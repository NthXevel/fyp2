"""
Feature engineering module for preparing data for XGBoost

Fixed feature set (no feature selection needed):
    Return, EMA_10, EMA_20, RSI_14, MACD_Hist, Volatility_14,
    Return_Lag_1, Return_Lag_3

Supports single-stock and multi-stock training.  When multiple symbols
are used, a 'symbol' column is carried through but NOT included as a
model feature — it's only used for tracking / debugging.
"""
import pandas as pd
import numpy as np


# The exact feature columns used for training / prediction
FEATURE_COLUMNS = [
    'Return', 'EMA_10', 'EMA_20', 'RSI_14', 'MACD_Hist',
    'Volatility_14', 'Return_Lag_1', 'Return_Lag_3',
]


class FeatureEngineer:
    def __init__(self):
        self.feature_cols = FEATURE_COLUMNS
    
    def create_features(self, df, symbol: str | None = None):
        """
        Create the fixed set of technical indicators from OHLCV data.

        Output columns (besides OHLCV):
            Return, EMA_10, EMA_20, RSI_14, MACD_Hist,
            Volatility_14, Return_Lag_1, Return_Lag_3, Target
        
        Args:
            df: DataFrame with columns [open, high, low, close, volume]
            symbol: Optional ticker name — added as a 'symbol' column
                    for multi-stock tracking.
            
        Returns:
            pd.DataFrame: DataFrame with engineered features + Target
        """
        df = df.copy()

        if symbol is not None:
            df['symbol'] = symbol
        
        # ── Return ───────────────────────────────────────────────────
        df['Return'] = df['close'].pct_change()

        # ── EMA 10 & 20 ─────────────────────────────────────────────
        df['EMA_10'] = df['close'].ewm(span=10).mean()
        df['EMA_20'] = df['close'].ewm(span=20).mean()

        # ── RSI 14 ──────────────────────────────────────────────────
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI_14'] = 100 - (100 / (1 + rs))

        # ── MACD Histogram ──────────────────────────────────────────
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        macd_line = ema_12 - ema_26
        macd_signal = macd_line.ewm(span=9).mean()
        df['MACD_Hist'] = macd_line - macd_signal

        # ── Volatility (14-period rolling std of returns) ───────────
        df['Volatility_14'] = df['Return'].rolling(window=14).std()

        # ── Lagged returns ──────────────────────────────────────────
        df['Return_Lag_1'] = df['Return'].shift(1)
        df['Return_Lag_3'] = df['Return'].shift(3)

        # ── Target (1 if next bar closes higher, else 0) ───────────
        df['Target'] = (df['close'].shift(-1) > df['close']).astype(int)
        
        return df
    
    def prepare_training_data(self, df):
        """
        Prepare data for XGBoost training by removing NaN values.
        Uses the fixed FEATURE_COLUMNS — no feature selection needed.
        
        Args:
            df: DataFrame with features and Target
            
        Returns:
            tuple: (X, y, feature_cols)
        """
        df = df.dropna(subset=self.feature_cols + ['Target'])
        
        X = df[self.feature_cols]
        y = df['Target']
        
        return X, y, self.feature_cols

    def prepare_multi_stock(self, dataframes: dict[str, pd.DataFrame]):
        """
        Build features for multiple symbols and stack them into one
        training dataset.  Features are computed PER SYMBOL so indicators
        don't bleed across tickers.

        Args:
            dataframes: {symbol: ohlcv_df} mapping

        Returns:
            tuple: (X, y, feature_cols) — concatenated across all symbols,
                   sorted chronologically.
        """
        all_frames = []
        for symbol, df in dataframes.items():
            featured = self.create_features(df, symbol=symbol)
            clean = featured.dropna(subset=self.feature_cols + ['Target'])
            all_frames.append(clean)

        combined = pd.concat(all_frames)
        combined.sort_index(inplace=True)

        X = combined[self.feature_cols]
        y = combined['Target']

        print(f"Multi-stock dataset: {len(X)} rows from "
              f"{len(dataframes)} symbols")
        return X, y, self.feature_cols
