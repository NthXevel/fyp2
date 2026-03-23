"""
Configuration module for loading environment variables and settings.

Supports both .env files (local) and Streamlit secrets (Community Cloud).
"""
import os
from dotenv import load_dotenv

load_dotenv()


def _get_secret(key: str, default: str | None = None) -> str | None:
    """Read from env vars first, then fall back to st.secrets (Streamlit Cloud)."""
    val = os.getenv(key)
    if val:
        return val
    try:
        import streamlit as st
        # Use bracket access with KeyError fallback; .get() is unreliable
        # on some Streamlit versions / when secrets file is missing.
        try:
            return st.secrets[key]
        except KeyError:
            return default
    except Exception:
        return default


# ── Performance Targets ──────────────────────────────────────────────
TARGET_ACCURACY = 0.55       # ≥ 55 %
TARGET_SHARPE = 0.5          # ≥ 0.5
TARGET_MAX_DRAWDOWN = 0.10   # ≤ 10 %

# Alpaca API Configuration
ALPACA_API_KEY = _get_secret('ALPACA_API_KEY')
ALPACA_SECRET_KEY = _get_secret('ALPACA_SECRET_KEY')
ALPACA_BASE_URL = _get_secret('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')

# Trading Configuration
STOCK_SYMBOL = _get_secret('STOCK_SYMBOL', 'BTC/USD')
INVESTMENT_AMOUNT = float(_get_secret('INVESTMENT_AMOUNT', '1000'))
MODEL_PATH = _get_secret('MODEL_PATH', 'models/saved/xgboost_model.pkl')

# ── Multi-stock training ─────────────────────────────────────────────
# Comma-separated list of symbols used for training.
# The model learns general patterns across all these tickers.
# At inference time it still trades STOCK_SYMBOL.
# Use slash notation for crypto pairs (e.g. BTC/USD).
TRAINING_SYMBOLS = [
    s.strip() for s in
    _get_secret('TRAINING_SYMBOLS',
             'BTC/USD,ETH/USD,SOL/USD,BNB/USD,ADA/USD,XRP/USD').split(',')
]
# Intervals for training data — 15m intraday only
TRAINING_INTERVALS = [s.strip() for s in _get_secret('TRAINING_INTERVALS', '15m').split(',')]
TRAINING_INTERVAL  = _get_secret('TRAINING_INTERVAL', '15m')   # default interval
TRAINING_DAYS      = int(_get_secret('TRAINING_DAYS', '59'))   # Yahoo max ~60 days for 15m
TRAINING_DAYS_15M  = int(_get_secret('TRAINING_DAYS_15M', '59'))  # Yahoo max ~60 days for 15m

# Confidence-based position sizing
CONFIDENCE_THRESHOLD = float(_get_secret('CONFIDENCE_THRESHOLD', '0.55'))
MIN_PCT_ALLOCATION = float(_get_secret('MIN_PCT_ALLOCATION', '0.10'))   # 10% of equity
MAX_PCT_ALLOCATION = float(_get_secret('MAX_PCT_ALLOCATION', '0.95'))   # 95% of equity (leave 5% buffer)

# Risk management -- per-order stop-loss / take-profit (based on order value)
STOP_LOSS_PCT = float(_get_secret('STOP_LOSS_PCT', '0.02'))     
TAKE_PROFIT_PCT = float(_get_secret('TAKE_PROFIT_PCT', '0.015'))   

# Data Configuration (live trading interval — kept for run_bot / data_fetcher)
DATA_INTERVAL = _get_secret('DATA_INTERVAL', '15m')  # 15-minute bars
DATA_DAYS = int(_get_secret('DATA_DAYS', '59'))       # Yahoo allows max 60 days for 15m

# Model Training Parameters
LOOKBACK_PERIOD = 60  # Number of days to look back for features
TEST_SIZE = 0.2
RANDOM_STATE = 42
# ── Crypto helpers ────────────────────────────────────────────────────
def is_crypto(symbol: str) -> bool:
    """Return True if the symbol looks like a crypto pair (e.g. BTC/USD)."""
    return '/' in symbol or symbol.upper().endswith('USD') and '-' in symbol

def yahoo_symbol(symbol: str) -> str:
    return symbol.replace('/', '-')

def alpaca_symbol(symbol: str) -> str:
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
