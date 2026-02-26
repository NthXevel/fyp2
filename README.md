# Quantitative Trading Bot - Apple Stock (AAPL)

A Python-based quantitative trading system using XGBoost machine learning model, Alpaca for trade execution, and Yahoo Finance for real-time market data.

## Project Structure

```
fyp2/
├── src/
│   ├── config.py              # Configuration and environment variables
│   ├── data_fetcher.py        # Yahoo Finance integration for data fetching
│   ├── feature_engineering.py # Technical indicators and feature creation
│   ├── model_trainer.py       # XGBoost model training and evaluation
│   ├── trading_executor.py    # Alpaca API integration for trade execution
│   └── __init__.py
├── data/                      # Directory for storing historical data
├── models/                    # Directory for saved trained models
├── train_model.py            # Script to train the XGBoost model
├── trading_bot.py            # Main trading bot for live execution
├── requirements.txt          # Python dependencies
├── .env.example              # Example environment variables
└── README.md                 # This file
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

1. Copy `.env.example` to `.env`:
```bash
copy .env.example .env
```

2. Get your Alpaca API keys:
   - Sign up at https://alpaca.markets and get your API keys

3. Edit `.env` file and add your Alpaca API keys:
```env
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
```

## Usage

### Step 1: Train the Model

First, train the XGBoost model using historical data:

```bash
python train_model.py
```

This script will:
- Fetch 1 year of historical Apple stock data from Yahoo Finance
- Engineer 20+ technical indicators
- Train XGBoost classifier
- Display model accuracy and feature importance
- Save the trained model to `models/xgboost_model.pkl`

### Step 2: Run the Trading Bot

Once the model is trained, start the trading bot:

```bash
python trading_bot.py
```

The bot will:
- Load the trained model
- Continuously monitor market conditions every hour
- Make buy/sell decisions based on model predictions
- Execute orders via Alpaca API
- Display account and position information

## Features

### Data Fetching
- Historical data from Yahoo Finance (no API key required)
- Real-time quotes from Yahoo Finance
- Ready for backtesting and live trading
- 1-minute to daily candle data

### Feature Engineering
- Technical indicators: SMA, EMA, MACD, RSI, Bollinger Bands
- Price-based features: returns, price range
- Volume-based features: volume ratio, volume SMA
- Momentum features
- Volatility measures

### Model
- XGBoost binary classifier
- Predicts daily up/down price movements
- Configurable hyperparameters in `config.py`

### Trading
- Paper trading support (default) for safe testing
- Market orders via Alpaca
- Position tracking and P&L monitoring
- Account balance management

## Configuration

Edit `src/config.py` to customize:

```python
LOOKBACK_PERIOD = 60          # Days for feature calculation
XGB_PARAMS = {
    'max_depth': 5,           # Tree depth
    'learning_rate': 0.1,     # Learning rate
    'n_estimators': 100,      # Number of trees
}
INVESTMENT_AMOUNT = 1000      # Dollar amount per trade
```

## Trading Strategy

The bot uses a simple strategy:
1. **Buy Signal**: Model predicts >60% probability of price increase
2. **Sell Signal**: Model predicts >60% probability of price decrease
3. **Hold**: Confidence below threshold or no existing position/cash

## Important Notes

⚠️ **Paper Trading**: The default configuration uses Alpaca's paper trading. To switch to live trading, change in `.env`:
```env
ALPACA_BASE_URL=https://api.alpaca.markets
```

⚠️ **Risk Disclaimer**: This system is for educational purposes. Trading stocks involves risk of loss. Always:
- Test thoroughly with paper trading first
- Use small position sizes
- Implement proper risk management
- Monitor the bot regularly

## Troubleshooting

### Model training fails
- Check Finnhub API key is valid
- Ensure internet connection
- Try with fewer days: `fetcher.get_historical_data(days=180)`

### Trading bot crashes
- Verify Alpaca API keys
- Check account has sufficient cash/buying power
- Ensure paper trading URL is correct

### No predictions
- Verify model file exists at `models/xgboost_model.pkl`
- Retrain the model if necessary

## Next Steps

To improve the system:
1. Add more technical indicators
2. Implement risk management (stop-loss, take-profit)
3. Add performance metrics and logging
4. Backtest strategy on historical data
5. Optimize hyperparameters
6. Add sentiment analysis

## Dependencies

- **XGBoost**: Machine learning library
- **Alpaca Trade API**: Broker API
- **Finnhub**: Market data provider
- **Pandas/NumPy**: Data processing
- **Scikit-learn**: ML utilities

## Author

Created for quantitative trading project (FYP2)

## License

Educational use only
