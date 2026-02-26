"""
Configuration module for loading environment variables and settings
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── Performance Targets ──────────────────────────────────────────────
TARGET_ACCURACY = 0.55       # ≥ 55 %
TARGET_SHARPE = 0.5          # ≥ 0.5
TARGET_MAX_DRAWDOWN = 0.10   # ≤ 10 %

# Alpaca API Configuration
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
ALPACA_BASE_URL = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')

# Trading Configuration
STOCK_SYMBOL = os.getenv('STOCK_SYMBOL', 'AAPL')
INVESTMENT_AMOUNT = float(os.getenv('INVESTMENT_AMOUNT', '1000'))
MODEL_PATH = os.getenv('MODEL_PATH', 'models/xgboost_model.pkl')

# Confidence-based position sizing
# Only enter a trade when confidence exceeds CONFIDENCE_THRESHOLD.
# A higher threshold filters out marginal signals, which lifts both
# accuracy and Sharpe while keeping drawdown in check.
CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', '0.55'))
MIN_INVESTMENT_AMOUNT = float(os.getenv('MIN_INVESTMENT_AMOUNT', '200'))
MAX_INVESTMENT_AMOUNT = float(os.getenv('MAX_INVESTMENT_AMOUNT', '2000'))

# Risk management -- per-order stop-loss / take-profit (based on order value)
STOP_LOSS_PCT = float(os.getenv('STOP_LOSS_PCT', '0.20'))      # 20 % per order
TAKE_PROFIT_PCT = float(os.getenv('TAKE_PROFIT_PCT', '0.20'))   # 20 % per order

# Data Configuration
DATA_INTERVAL = os.getenv('DATA_INTERVAL', '15m')  # 15-minute bars
DATA_DAYS = int(os.getenv('DATA_DAYS', '59'))       # Yahoo allows max 60 days for 15m

# Model Training Parameters
LOOKBACK_PERIOD = 60  # Number of days to look back for features
TEST_SIZE = 0.2
RANDOM_STATE = 42
XGB_PARAMS = {
    'max_depth': 4,
    'learning_rate': 0.03,
    'n_estimators': 300,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'random_state': RANDOM_STATE,
}
