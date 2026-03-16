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
from datetime import datetime, timezone

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.trainer import ModelTrainer
from utils.data_fetcher import DataFetcher
from strategies.feature_engineering import FeatureEngineer
from strategies.signal_generator import SignalGenerator
from execution.alpaca_executor import TradingExecutor
from config.settings import (
    STOCK_SYMBOL, INVESTMENT_AMOUNT, LOOKBACK_PERIOD, DATA_INTERVAL,
    CONFIDENCE_THRESHOLD, MIN_PCT_ALLOCATION, MAX_PCT_ALLOCATION,
    STOP_LOSS_PCT, TAKE_PROFIT_PCT,
    TARGET_ACCURACY, TARGET_SHARPE, TARGET_MAX_DRAWDOWN,
    is_crypto,
)
from utils.db_connector import init_database, insert_prediction


class TradingBot:
    def __init__(self):
        self.trainer = ModelTrainer()
        self.fetcher = DataFetcher()
        self.engineer = FeatureEngineer()
        self.executor = TradingExecutor()
        self.signal = SignalGenerator()
        self.sell_cooldown_bars_after_buy = 8
        self.sell_cooldown_remaining = 0
        init_database()

        # Risk-management state
        self._is_crypto = is_crypto(STOCK_SYMBOL)

        # Recover entry price from existing position (survives restarts)
        existing = self.executor.get_position()
        if existing and existing['qty'] > 0:
            self.entry_price = existing['avg_entry_price']
            print(f"Recovered existing position: {existing['qty']} @ ${self.entry_price:.2f}")
        else:
            self.entry_price = None

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
            tuple[pd.DataFrame, float | None, float | None]:
                (latest_features, macro_sma_4h, latest_close)
        """
        # Need enough bars for indicator warmup; fetch extra days
        df = self.fetcher.get_historical_data(days=min(LOOKBACK_PERIOD + 10, 59))
        if df is None:
            return None

        df = self.engineer.create_features(df)
        feature_cols = self.trainer.feature_cols

        # Get the latest row
        latest = df[feature_cols].iloc[-1:]
        macro_sma_4h = df['Macro_SMA_4H'].iloc[-1] if 'Macro_SMA_4H' in df.columns else None
        latest_close = df['close'].iloc[-1] if 'close' in df.columns else None
        return latest, macro_sma_4h, latest_close

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
            # Skip buying if already holding a position
            if position and position['qty'] > 0:
                print(f"Already holding {position['qty']} — skipping BUY")
                return

            quote = self.fetcher.get_realtime_quote()
            if quote:
                current_price = quote['c']
                equity = float(account['equity'])
                qty, investment = self.signal.calculate_position_size(
                    confidence, current_price, equity, fractional=self._is_crypto
                )

                print(f"Confidence: {confidence:.2%} -> Investment: ${investment:.2f} -> Qty: {qty} shares")

                if qty > 0 and float(account['buying_power']) > qty * current_price:
                    self.executor.place_buy_order(
                        qty,
                        confidence=confidence,
                        investment=investment,
                        capital=float(account['equity']),
                    )
                    self.entry_price = current_price
                    self.sell_cooldown_remaining = self.sell_cooldown_bars_after_buy
                else:
                    print(f"Insufficient buying power. Available: ${account['buying_power']}")

        elif action == 'sell':
            if position and position['qty'] > 0:
                sell_qty = position['qty'] if self._is_crypto else int(position['qty'])
                self.executor.place_sell_order(
                    sell_qty,
                    confidence=confidence,
                    capital=float(account['equity']),
                )
                self.entry_price = None
                self.sell_cooldown_remaining = 0
            else:
                print("No position to sell")

    def run(self, check_interval=300, ml_interval=900):
        """
        Run the trading bot in a loop.

        The bot wakes every *check_interval* seconds to evaluate SL/TP
        against the real-time quote.  The heavier ML prediction cycle only
        runs every *ml_interval* seconds (aligned to 15-min bars).

        Args:
            check_interval: Seconds between risk-management checks (default: 300 = 5 min)
            ml_interval:    Seconds between ML prediction cycles     (default: 900 = 15 min)
        """
        print("\n" + "=" * 60)
        print(f"Starting Trading Bot - {STOCK_SYMBOL}")
        print(f"  Risk-management poll : every {check_interval}s")
        print(f"  ML prediction cycle  : every {ml_interval}s")
        print("=" * 60)

        last_ml_run = 0  # epoch – forces ML to run on the first iteration

        try:
            while True:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"\n[{timestamp}] Checking market conditions...")

                # ── 1. Real-time quote & position (cheap, every wake-up) ──
                quote = self.fetcher.get_realtime_quote()
                if quote:
                    print(f"  {STOCK_SYMBOL}  Last: ${quote['c']:.2f}  Vol: {int(quote['v']):,}")

                account = self.executor.get_account_info()
                position = self.executor.get_position()
                if account:
                    print(f"  Cash: ${account['cash']:.2f}  Equity: ${account['equity']:.2f}")
                if position:
                    print(f"  Position: {position['qty']} shares  P&L: ${position['unrealized_pl']:.2f}")

                # ── 2. Check stop-loss / take-profit FIRST ────────────────
                if position and position['qty'] > 0 and quote:
                    sl_tp_action = self.signal.check_stop_loss_take_profit(
                        self.entry_price, quote['c']
                    )
                    if sl_tp_action == 'sell':
                        self.execute_trade('sell', confidence=0.0)
                        print(f"Next check in {check_interval}s...")
                        time.sleep(check_interval)
                        continue

                # ── 3. ML prediction (only every ml_interval) ─────────────
                now = time.time()
                if now - last_ml_run >= ml_interval:
                    last_ml_run = now

                    features_data = self.get_latest_features()
                    if features_data is None:
                        print("Failed to fetch latest data. Retrying...")
                        time.sleep(60)
                        continue
                    features, macro_sma_4h, latest_close = features_data

                    # Make prediction
                    probabilities = self.trainer.predict(features)
                    action, confidence = self.signal.decide_trade(probabilities)

                    # Macro trend filter: only buy when price > 4h SMA
                    if action == 'buy':
                        trend_price = quote['c'] if quote else latest_close
                        if macro_sma_4h is None or trend_price is None:
                            action = 'hold'
                            print("Trend filter unavailable (missing Macro_SMA_4H) -> HOLD")
                        elif trend_price <= macro_sma_4h:
                            action = 'hold'
                            print(f"Trend filter blocked BUY: price ${trend_price:.2f} <= 4h SMA ${macro_sma_4h:.2f}")

                    # Cooldown only suppresses model-driven SELL signals.
                    # SL/TP executed above bypasses this guard.
                    if action == 'sell' and self.sell_cooldown_remaining > 0:
                        print(f"Cooldown active ({self.sell_cooldown_remaining} bars remaining) -> ignoring model SELL")
                        action = 'hold'

                    print(f"Decision: {action.upper()} (confidence: {confidence:.2%})")

                    try:
                        insert_prediction(
                            prediction_time=datetime.now(timezone.utc),
                            symbol=STOCK_SYMBOL,
                            signal=action,
                            prob_up=float(probabilities[0][1]),
                            prob_down=float(probabilities[0][0]),
                            confidence=float(confidence),
                            timeframe=DATA_INTERVAL,
                            model_name="xgboost",
                        )
                    except Exception as exc:
                        print(f"Warning: prediction logging failed: {exc}")

                    # Execute trade
                    if action in ['buy', 'sell']:
                        self.execute_trade(action, confidence)

                    # Count down sell cooldown one bar at a time while holding.
                    position_after = self.executor.get_position()
                    if position_after and position_after['qty'] > 0 and self.sell_cooldown_remaining > 0:
                        self.sell_cooldown_remaining -= 1
                else:
                    print("  (SL/TP check only — next ML prediction in "
                          f"{int(ml_interval - (now - last_ml_run))}s)")

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

    # Risk-management poll every 5 min; ML prediction every 15 min
    bot.run(check_interval=300, ml_interval=900)


if __name__ == "__main__":
    main()
