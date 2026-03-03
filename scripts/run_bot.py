"""
Main trading bot script for real-time trading

Targets:
    • Accuracy  ≥ 55 %
    • Sharpe    ≥ 0.5
    • Max DD    ≤ 10 %
"""
import time
import sys
import os
from datetime import datetime

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.trainer import ModelTrainer
from utils.data_fetcher import DataFetcher
from strategies.feature_engineering import FeatureEngineer
from strategies.signal_generator import SignalGenerator
from execution.alpaca_executor import TradingExecutor
from monitoring.logger import TradeLogger
from config.settings import (
    STOCK_SYMBOL, INVESTMENT_AMOUNT, LOOKBACK_PERIOD, DATA_INTERVAL,
    CONFIDENCE_THRESHOLD, MIN_INVESTMENT_AMOUNT, MAX_INVESTMENT_AMOUNT,
    STOP_LOSS_PCT, TAKE_PROFIT_PCT,
    TARGET_ACCURACY, TARGET_SHARPE, TARGET_MAX_DRAWDOWN,
    is_crypto,
)


class TradingBot:
    def __init__(self):
        self.trainer = ModelTrainer()
        self.fetcher = DataFetcher()
        self.engineer = FeatureEngineer()
        self.executor = TradingExecutor()
        self.signal = SignalGenerator()
        self.logger = TradeLogger()

        # Risk-management state
        self.entry_price = None   # track buy price for SL/TP
        self._is_crypto = is_crypto(STOCK_SYMBOL)

        # Load trained model
        if not self.trainer.load():
            print("Error: Model not found. Please train the model first using: python -m scripts.train")
            sys.exit(1)

        print(f"Trading Bot initialized for {STOCK_SYMBOL}")
        print(f"Targets: Acc >= {TARGET_ACCURACY:.0%}  |  "
              f"Sharpe >= {TARGET_SHARPE}  |  Max DD <= {TARGET_MAX_DRAWDOWN:.0%}")
        print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
        print(f"Stop-loss: {STOP_LOSS_PCT:.1%}  |  Take-profit: {TAKE_PROFIT_PCT:.1%}")

    def get_latest_features(self):
        """
        Fetch latest 15m data and compute features for prediction

        Returns:
            pd.DataFrame: Latest features for prediction
        """
        # Need enough bars for indicator warmup; fetch extra days
        df = self.fetcher.get_historical_data(days=min(LOOKBACK_PERIOD + 10, 59))
        if df is None:
            return None

        df = self.engineer.create_features(df)
        feature_cols = self.trainer.feature_cols

        # Get the latest row
        latest = df[feature_cols].iloc[-1:]
        return latest

    def execute_trade(self, action, confidence=0.5):
        """
        Execute trade based on decision. The model's confidence determines
        how many shares to buy (scaled between MIN and MAX investment).

        Args:
            action: 'buy' or 'sell'
            confidence: Model prediction probability (0.0 - 1.0)
        """
        account = self.executor.get_account_info()
        if not account:
            print("Could not retrieve account info")
            return

        position = self.executor.get_position()

        if action == 'buy':
            quote = self.fetcher.get_realtime_quote()
            if quote:
                current_price = quote['c']
                qty, investment = self.signal.calculate_position_size(
                    confidence, current_price, fractional=self._is_crypto
                )

                print(f"Confidence: {confidence:.2%} -> Investment: ${investment:.2f} -> Qty: {qty} shares")

                if qty > 0 and float(account['buying_power']) > qty * current_price:
                    self.executor.place_buy_order(qty)
                    self.entry_price = current_price
                    self.logger.log('BUY', STOCK_SYMBOL, qty, current_price,
                                    confidence, investment)
                else:
                    print(f"Insufficient buying power. Available: ${account['buying_power']}")

        elif action == 'sell':
            if position and float(position['qty']) > 0:
                sell_qty = float(position['qty']) if self._is_crypto else int(position['qty'])
                self.executor.place_sell_order(sell_qty)
                self.logger.log('SELL', STOCK_SYMBOL, sell_qty,
                                position['current_price'], confidence)
                self.entry_price = None
            else:
                print("No position to sell")

    def run(self, check_interval=3600):
        """
        Run the trading bot in a loop

        Args:
            check_interval: Seconds between checks (default: 1 hour)
        """
        print("\n" + "=" * 60)
        print(f"Starting Trading Bot - {STOCK_SYMBOL}")
        print("=" * 60)

        try:
            while True:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"\n[{timestamp}] Checking market conditions...")

                # Get latest features
                features = self.get_latest_features()
                if features is None:
                    print("Failed to fetch latest data. Retrying...")
                    time.sleep(60)
                    continue

                # Fetch real-time stock info
                quote = self.fetcher.get_realtime_quote()
                if quote:
                    print(f"  {STOCK_SYMBOL}  Last: ${quote['c']:.2f}  Vol: {int(quote['v']):,}")

                # Account & position snapshot
                account = self.executor.get_account_info()
                position = self.executor.get_position()
                if account:
                    print(f"  Cash: ${account['cash']:.2f}  Equity: ${account['equity']:.2f}")
                if position:
                    print(f"  Position: {position['qty']} shares  P&L: ${position['unrealized_pl']:.2f}")

                # ── Check stop-loss / take-profit first ──────────────
                position = self.executor.get_position()
                if position and quote:
                    sl_tp_action = self.signal.check_stop_loss_take_profit(
                        self.entry_price, quote['c']
                    )
                    if sl_tp_action == 'sell':
                        self.execute_trade('sell', confidence=0.0)
                        print(f"Next check in {check_interval}s...")
                        time.sleep(check_interval)
                        continue

                # Make prediction
                probabilities = self.trainer.predict(features)
                action, confidence = self.signal.decide_trade(probabilities)

                print(f"Decision: {action.upper()} (confidence: {confidence:.2%})")

                # Execute trade
                if action in ['buy', 'sell']:
                    self.execute_trade(action, confidence)

                # Wait for next check
                print(f"Next check in {check_interval}s...")
                time.sleep(check_interval)

        except KeyboardInterrupt:
            print("\nTrading bot stopped by user")
        except Exception as e:
            print(f"Error in trading loop: {e}")


def main():
    bot = TradingBot()

    # Print account info
    account = bot.executor.get_account_info()
    if account:
        print("\nAccount Information:")
        print(f"  Cash: ${account['cash']:.2f}")
        print(f"  Portfolio Value: ${account['portfolio_value']:.2f}")
        print(f"  Buying Power: ${account['buying_power']:.2f}")

    # Run the bot — check every 15 minutes to match the 15m bar interval
    bot.run(check_interval=900)


if __name__ == "__main__":
    main()
