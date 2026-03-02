# Quantitative Trading Bot - Apple Stock (AAPL)

A Python-based quantitative trading system using XGBoost machine learning model, Alpaca for trade execution, and Yahoo Finance for real-time market data.

## Project Structure

```
fyp2/
├── config/                         # Configuration files
│   ├── __init__.py
│   └── settings.py                 # Environment variables & hyperparameters
├── data/                           # Raw & processed market data
│   └── AAPL_15m.csv
├── models/                         # Trained XGBoost models
│   ├── __init__.py
│   ├── trainer.py                  # XGBoost training & evaluation
│   └── saved/                      # Serialised .pkl model files
├── strategies/                     # Signal generation & position sizing
│   ├── __init__.py
│   ├── feature_engineering.py      # Technical indicators & feature creation
│   └── signal_generator.py         # Buy/sell signals, position sizing, SL/TP
├── execution/                      # Alpaca order management
│   ├── __init__.py
│   └── alpaca_executor.py          # Alpaca API integration
├── backtesting/                    # Backtesting engine
│   ├── __init__.py
│   └── engine.py                   # Historical backtest with SL/TP & metrics
├── monitoring/                     # Dashboard & alerting
│   ├── __init__.py
│   ├── dashboard.py                # Console display for quotes & account
│   └── logger.py                   # Trade logging to CSV
├── utils/                          # Shared utilities
│   ├── __init__.py
│   └── data_fetcher.py             # Yahoo Finance data fetching
├── scripts/                        # Entry point scripts
│   ├── __init__.py
│   ├── run_bot.py                  # Live trading bot
│   ├── train.py                    # Model training
│   └── download_data.py            # Data downloader
├── reports/                        # Generated reports & plots
├── experiments.ipynb               # EDA, tuning, backtesting notebook
├── requirements.txt                # Python dependencies
└── README.md
```

## Setup Instructions

### 1. Clone/Create Environment

```bash
cd c:\Users\jieha\fyp2
```

### 2. Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API Keys

1. Create a `.env` file in the project root:

2. Get your Alpaca API keys:
   - Sign up at https://alpaca.markets and get your API keys

3. Add your Alpaca API keys to `.env`:
```env
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
```

## Usage

### Download Market Data

```bash
python -m scripts.download_data
python -m scripts.download_data --symbol TSLA
python -m scripts.download_data --symbols AAPL TSLA MSFT
```

### Train the Model

```bash
python -m scripts.train
```

This script will:
- Fetch historical Apple stock data (or load from local CSV)
- Engineer 8 technical indicator features
- Train XGBoost classifier with time-series CV hyperparameter search
- Display model accuracy and feature importance
- Save the trained model to `models/saved/`

### Run the Trading Bot

```bash
python -m scripts.run_bot
```

The bot will:
- Load the trained model
- Continuously monitor market conditions every 15 minutes
- Make buy/sell decisions based on model predictions
- Execute orders via Alpaca API
- Display account and position information

## Features

### Data Fetching (`utils/`)
- Historical data from Yahoo Finance (no API key required)
- Real-time quotes from Yahoo Finance
- 1-minute to daily candle data

### Feature Engineering (`strategies/`)
- Technical indicators: EMA, MACD, RSI
- Price-based features: returns, lagged returns
- Volatility measures (14-period rolling std)

### Signal Generation (`strategies/`)
- Confidence-based position sizing ($200 – $20,000)
- Per-order stop-loss (20%) and take-profit (20%)
- Threshold filtering for marginal signals

### Model (`models/`)
- XGBoost binary classifier
- Predicts 15-minute bar up/down movements
- Time-series cross-validation hyperparameter tuning

### Execution (`execution/`)
- Paper trading support (default) for safe testing
- Market orders via Alpaca
- Position tracking and P&L monitoring

### Backtesting (`backtesting/`)
- Historical backtest engine with confidence-based sizing
- Stop-loss and take-profit simulation
- Sharpe ratio, max drawdown, win rate metrics

### Monitoring (`monitoring/`)
- Real-time console dashboard for quotes and account
- Trade logging to CSV for audit and analysis

## Configuration

Edit `config/settings.py` to customize:

```python
LOOKBACK_PERIOD = 60          # Days for feature calculation
CONFIDENCE_THRESHOLD = 0.55   # Minimum confidence to trade
STOP_LOSS_PCT = 0.20          # 20% per-order stop-loss
TAKE_PROFIT_PCT = 0.20        # 20% per-order take-profit
XGB_PARAMS = {
    'max_depth': 4,
    'learning_rate': 0.03,
    'n_estimators': 300,
}
```

## Trading Strategy

1. **Buy Signal**: Model predicts >55% probability of price increase
2. **Sell Signal**: Model predicts >55% probability of price decrease
3. **Stop-Loss**: Individual order drops 20% from entry → automatic sell
4. **Take-Profit**: Individual order gains 20% from entry → automatic sell
5. **Position Sizing**: Investment scaled $200–$20,000 based on confidence

## Performance Targets

| Metric       | Target  |
|-------------|---------|
| Accuracy    | ≥ 55%   |
| Sharpe Ratio| ≥ 0.5   |
| Max Drawdown| ≤ 10%   |

## Important Notes

⚠️ **Paper Trading**: The default configuration uses Alpaca's paper trading. To switch to live trading, change in `.env`:
```env
ALPACA_BASE_URL=https://api.alpaca.markets
```

⚠️ **Risk Disclaimer**: This system is for educational purposes. Trading stocks involves risk of loss. Always test thoroughly with paper trading first.

## Dependencies

- **XGBoost**: Machine learning library
- **Alpaca Trade API**: Broker API
- **yfinance**: Yahoo Finance market data
- **Pandas/NumPy**: Data processing
- **Scikit-learn**: ML utilities

## Author

Created for quantitative trading project (FYP2)

## License

Educational use only
