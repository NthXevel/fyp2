"""
Feature engineering module for preparing data for XGBoost

All features are **stationary** (relative / normalised) so that tree
splits remain valid across different price regimes.

Feature groups:
    Price-action : Return, EMA_10, EMA_20, Volatility_14,
                   Return_Lag_1, Return_Lag_3
    Momentum     : RSI_14 (z-scored RSI), MACD_Hist (normalised)

Target:
    3-day forward return > +0.5 % → class 1 (Buy), else 0

Supports single-stock and multi-stock training.  When multiple symbols
are used, a 'symbol' column is carried through but NOT included as a
model feature — it's only used for tracking / debugging.
"""
import pandas as pd
import numpy as np

# How many bars ahead the target looks & minimum move to count as "up"
FORWARD_BARS = 3
TARGET_THRESHOLD = 0.005     # 0.5 %
MACRO_SMA_WINDOW_BARS = 16   # 4h SMA on 15m bars

# The exact feature columns used for training / prediction
FEATURE_COLUMNS = [
    # Price-action (stationary)
    'Return', 'EMA_10', 'EMA_20',
    'Volatility_14', 'Return_Lag_1', 'Return_Lag_3',
    # Momentum (normalised)
    'RSI_14', 'MACD_Hist', 'MACD_Line_Norm',
]


class FeatureEngineer:
    def __init__(self):
        self.feature_cols = FEATURE_COLUMNS

    def create_features(self, df, symbol: str | None = None):
        df = df.copy()

        if symbol is not None:
            df['symbol'] = symbol
        
        # ── Return ───────────────────────────────────────────────────
        df['Return'] = df['close'].pct_change()

        # ── EMA 10 & 20 (FIXED: Converted to Relative Distance) ──────
        df['EMA_10'] = (df['close'] / df['close'].ewm(span=10).mean()) - 1
        df['EMA_20'] = (df['close'] / df['close'].ewm(span=20).mean()) - 1

        # ── RSI 14 (FIXED: Normalized as Z-Score) ────────────────────
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        raw_rsi = 100 - (100 / (1 + rs))
        # Normalize RSI so it's stationary
        df['RSI_14'] = (raw_rsi - raw_rsi.rolling(14).mean()) / raw_rsi.rolling(14).std()

        # ── MACD Histogram ──────────────────────────────────────────
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        macd_line = ema_12 - ema_26
        macd_signal = macd_line.ewm(span=9).mean()
        # Convert MACD to a percentage of price so it scales across years
        df['MACD_Hist'] = (macd_line - macd_signal) / df['close']
        # "Zero Line" position — normalized by price so it works across assets
        df['MACD_Line_Norm'] = macd_line / df['close']

        # ── Volatility (14-period rolling std of returns) ───────────
        df['Volatility_14'] = df['Return'].rolling(window=14).std()

        # ── Lagged returns ──────────────────────────────────────────
        df['Return_Lag_1'] = df['Return'].shift(1)
        df['Return_Lag_3'] = df['Return'].shift(3)

        # ── Macro trend filter helper (not used as model feature) ───
        # 4-hour SMA derived from 15m bars for high-timeframe trend gating.
        df['Macro_SMA_4H'] = df['close'].rolling(window=MACRO_SMA_WINDOW_BARS).mean()

        # ── Target (FIXED: Predict a larger 3-day move to beat fees) ─
        # Transaction costs in your engine are 0.1% per trade (0.2% round trip).
        # If we only predict a 1-day move, fees eat all our profits.
        # Let's predict if the stock will be up more than 0.5% over the next 3 days.
        future_return = (df['close'].shift(-3) / df['close']) - 1
        df['Target'] = (future_return > 0.008).astype(int) 
        
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
