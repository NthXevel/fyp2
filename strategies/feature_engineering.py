"""
Feature engineering module for preparing data for XGBoost

All features are **stationary** (relative / normalised) so that tree
splits remain valid across different price regimes.

Feature groups:
    Price-action : Return, Dist_EMA_10, Dist_EMA_20, Volatility_14,
                   Return_Lag_1, Return_Lag_3
    Momentum     : RSI_Z (z-scored RSI), MACD_Hist_Norm (normalised)
    Volume       : OBV_Pct, VWAP_Dist, Volume_Ratio
    Candle       : Body_Ratio, Upper_Shadow, Lower_Shadow

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

# The exact feature columns used for training / prediction
FEATURE_COLUMNS = [
    # Price-action (stationary)
    'Return', 'Dist_EMA_10', 'Dist_EMA_20',
    'Volatility_14', 'Return_Lag_1', 'Return_Lag_3',
    # Momentum (normalised)
    'RSI_Z', 'MACD_Hist_Norm',
    # Volume
    'OBV_Pct', 'VWAP_Dist', 'Volume_Ratio',
    # Candle shape
    'Body_Ratio', 'Upper_Shadow', 'Lower_Shadow',
]


class FeatureEngineer:
    def __init__(self):
        self.feature_cols = FEATURE_COLUMNS

    def create_features(self, df, symbol: str | None = None):
        """
        Create stationary, diversity-rich features from OHLCV data.

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

        # ── EMA distances (stationary — Fix #1) ─────────────────────
        df['Dist_EMA_10'] = df['close'] / df['close'].ewm(span=10).mean() - 1
        df['Dist_EMA_20'] = df['close'] / df['close'].ewm(span=20).mean() - 1

        # ── RSI 14 → Z-scored (Fix #3) ──────────────────────────────
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        rsi_mean = rsi.rolling(20).mean()
        rsi_std  = rsi.rolling(20).std()
        df['RSI_Z'] = (rsi - rsi_mean) / rsi_std.replace(0, np.nan)

        # ── MACD Histogram — normalised by price (Fix #1) ───────────
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        macd_line   = ema_12 - ema_26
        macd_signal = macd_line.ewm(span=9).mean()
        df['MACD_Hist_Norm'] = (macd_line - macd_signal) / df['close']

        # ── Volatility (14-period rolling std of returns) ───────────
        df['Volatility_14'] = df['Return'].rolling(window=14).std()

        # ── Lagged returns ──────────────────────────────────────────
        df['Return_Lag_1'] = df['Return'].shift(1)
        df['Return_Lag_3'] = df['Return'].shift(3)

        # ── Volume features (Fix #3 — feature diversity) ───────────
        # OBV percentage change (stationary volume momentum)
        obv = (np.sign(df['close'].diff()) * df['volume']).cumsum()
        df['OBV_Pct'] = obv.pct_change(5)  # 5-bar OBV momentum

        # VWAP distance (intraday-friendly)
        cum_vol   = df['volume'].rolling(20).sum()
        cum_vp    = (df['close'] * df['volume']).rolling(20).sum()
        vwap      = cum_vp / cum_vol.replace(0, np.nan)
        df['VWAP_Dist'] = df['close'] / vwap - 1

        # Volume ratio vs 20-bar average
        df['Volume_Ratio'] = df['volume'] / df['volume'].rolling(20).mean()

        # ── Candle-body features ────────────────────────────────────
        bar_range = (df['high'] - df['low']).replace(0, np.nan)
        df['Body_Ratio']   = (df['close'] - df['open']) / bar_range
        df['Upper_Shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / bar_range
        df['Lower_Shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / bar_range

        # ── Target — smoothed forward return with threshold (Fix #2)
        future_return = df['close'].shift(-FORWARD_BARS) / df['close'] - 1
        df['Target'] = (future_return > TARGET_THRESHOLD).astype(int)

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
