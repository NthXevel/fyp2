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
STOCK_SYMBOL = os.getenv('STOCK_SYMBOL', 'BTC/USD')
INVESTMENT_AMOUNT = float(os.getenv('INVESTMENT_AMOUNT', '1000'))
MODEL_PATH = os.getenv('MODEL_PATH', 'models/saved/xgboost_model.pkl')

# ── Multi-stock training ─────────────────────────────────────────────
# Comma-separated list of symbols used for training.
# The model learns general patterns across all these tickers.
# At inference time it still trades STOCK_SYMBOL.
# Use slash notation for crypto pairs (e.g. BTC/USD).
TRAINING_SYMBOLS = [
    s.strip() for s in
    os.getenv('TRAINING_SYMBOLS',
             'BTC/USD,ETH/USD,SOL/USD,BNB/USD,ADA/USD,XRP/USD').split(',')
]
# Intervals for training data — 15m intraday only
TRAINING_INTERVALS = [s.strip() for s in os.getenv('TRAINING_INTERVALS', '15m').split(',')]
TRAINING_INTERVAL  = os.getenv('TRAINING_INTERVAL', '15m')   # default interval
TRAINING_DAYS      = int(os.getenv('TRAINING_DAYS', '59'))   # Yahoo max ~60 days for 15m
TRAINING_DAYS_15M  = int(os.getenv('TRAINING_DAYS_15M', '59'))  # Yahoo max ~60 days for 15m

# Confidence-based position sizing
CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', '0.65'))
MIN_PCT_ALLOCATION = float(os.getenv('MIN_PCT_ALLOCATION', '0.10'))   # 10% of equity
MAX_PCT_ALLOCATION = float(os.getenv('MAX_PCT_ALLOCATION', '0.95'))   # 95% of equity (leave 5% buffer)

# Risk management -- per-order stop-loss / take-profit (based on order value)
STOP_LOSS_PCT = float(os.getenv('STOP_LOSS_PCT', '0.02'))     
TAKE_PROFIT_PCT = float(os.getenv('TAKE_PROFIT_PCT', '0.03'))   

# Data Configuration (live trading interval — kept for run_bot / data_fetcher)
DATA_INTERVAL = os.getenv('DATA_INTERVAL', '15m')  # 15-minute bars
DATA_DAYS = int(os.getenv('DATA_DAYS', '59'))       # Yahoo allows max 60 days for 15m

# Model Training Parameters
LOOKBACK_PERIOD = 60  # Number of days to look back for features
TEST_SIZE = 0.2
RANDOM_STATE = 42
# ── Crypto helpers ────────────────────────────────────────────────────
def is_crypto(symbol: str) -> bool:
    """Return True if the symbol looks like a crypto pair (e.g. BTC/USD)."""
    return '/' in symbol or symbol.upper().endswith('USD') and '-' in symbol

def yahoo_symbol(symbol: str) -> str:
    """Convert internal symbol notation to Yahoo Finance ticker.
    BTC/USD  → BTC-USD
    ETH/USD  → ETH-USD
    AAPL     → AAPL  (unchanged)
    """
    return symbol.replace('/', '-')

def alpaca_symbol(symbol: str) -> str:
    """Convert internal symbol notation to Alpaca API format.
    BTC-USD  → BTC/USD
    BTC/USD  → BTC/USD  (unchanged)
    AAPL     → AAPL     (unchanged)
    """
    if '-' in symbol and symbol.upper().endswith('USD'):
        return symbol.replace('-', '/')
    return symbol

def safe_filename(symbol: str) -> str:
    """Return a filesystem-safe version of a symbol (strip / and -)."""
    return symbol.replace('/', '').replace('-', '')

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
