"""
Main trading bot script for real-time trading

Targets:
    • Accuracy  ≥ 55 %
    • Sharpe    ≥ 0.5
    • Max DD    ≤ 10 %
"""
import time
import sys
from datetime import datetime
from src.model_trainer import ModelTrainer
from src.data_fetcher import DataFetcher
from src.feature_engineering import FeatureEngineer
from src.trading_executor import TradingExecutor
from src.config import (
    STOCK_SYMBOL, INVESTMENT_AMOUNT, LOOKBACK_PERIOD, DATA_INTERVAL,
    CONFIDENCE_THRESHOLD, MIN_INVESTMENT_AMOUNT, MAX_INVESTMENT_AMOUNT,
    STOP_LOSS_PCT, TAKE_PROFIT_PCT,
    TARGET_ACCURACY, TARGET_SHARPE, TARGET_MAX_DRAWDOWN,
)


class TradingBot:
    def __init__(self):
        self.trainer = ModelTrainer()
        self.fetcher = DataFetcher()
        self.engineer = FeatureEngineer()
        self.executor = TradingExecutor()
        
        # Risk-management state
        self.entry_price = None   # track buy price for SL/TP
        
        # Load trained model
        if not self.trainer.load():
            print("Error: Model not found. Please train the model first using train_model.py")
            sys.exit(1)
        
        print(f"Trading Bot initialized for {STOCK_SYMBOL}")
        print(f"Targets: Acc ≥ {TARGET_ACCURACY:.0%}  |  "
              f"Sharpe ≥ {TARGET_SHARPE}  |  Max DD ≤ {TARGET_MAX_DRAWDOWN:.0%}")
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
    
    def decide_trade(self, features):
        """
        Decide whether to buy or sell based on model prediction
        
        Args:
            features: Feature vector for prediction
            
        Returns:
            str: 'buy', 'sell', or 'hold'
        """
        probabilities = self.trainer.predict(features)
        prob_down = probabilities[0][0]  # Probability of down
        prob_up = probabilities[0][1]    # Probability of up
        
        threshold = CONFIDENCE_THRESHOLD
        
        if prob_up > threshold:
            return 'buy', prob_up
        elif prob_down > threshold:
            return 'sell', prob_down
        else:
            return 'hold', max(prob_up, prob_down)
    
    def execute_trade(self, action, confidence=0.5):
        """
        Execute trade based on decision. The model's confidence determines
        how many shares to buy (scaled between MIN and MAX investment).
        Stop-loss and take-profit are enforced at the next check cycle.

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
            # Scale investment amount based on model confidence
            scale = (confidence - CONFIDENCE_THRESHOLD) / (1.0 - CONFIDENCE_THRESHOLD)
            scale = max(0.0, min(1.0, scale))
            investment = MIN_INVESTMENT_AMOUNT + scale * (MAX_INVESTMENT_AMOUNT - MIN_INVESTMENT_AMOUNT)

            quote = self.fetcher.get_realtime_quote()
            if quote:
                current_price = quote['c']
                qty = int(investment / current_price)

                print(f"Confidence: {confidence:.2%} → Investment: ${investment:.2f} → Qty: {qty} shares")

                if qty > 0 and float(account['buying_power']) > qty * current_price:
                    self.executor.place_buy_order(qty)
                    self.entry_price = current_price  # record for SL/TP
                else:
                    print(f"Insufficient buying power. Available: ${account['buying_power']}")
            
        elif action == 'sell':
            if position and int(position['qty']) > 0:
                self.executor.place_sell_order(int(position['qty']))
                self.entry_price = None
            else:
                print("No position to sell")

    def _check_stop_loss_take_profit(self, current_price):
        """Return 'sell' if SL or TP hit, else None.
        
        SL/TP are evaluated on the *order* entry price, not total portfolio.
        When the order drops 20 % from entry → stop-loss.
        When the order gains 20 % from entry → take-profit.
        """
        if self.entry_price is None:
            return None
        change = (current_price - self.entry_price) / self.entry_price
        if change <= -STOP_LOSS_PCT:
            print(f"⚠ STOP-LOSS triggered ({change:.2%} vs -{STOP_LOSS_PCT:.1%}) [per-order]")
            return 'sell'
        if change >= TAKE_PROFIT_PCT:
            print(f"✓ TAKE-PROFIT triggered ({change:.2%} vs +{TAKE_PROFIT_PCT:.1%}) [per-order]")
            return 'sell'
        return None
    
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
                
                # Fetch and display real-time stock info
                quote = self.fetcher.get_realtime_quote()
                if quote:
                    print(f"\n{'─'*50}")
                    print(f"  {STOCK_SYMBOL}  Real-Time Quote")
                    print(f"{'─'*50}")
                    print(f"  Last Price:  ${quote['c']:.2f}")
                    if quote.get('day_change') is not None:
                        sign = '+' if quote['day_change'] >= 0 else ''
                        print(f"  Day Change:  {sign}${quote['day_change']:.2f} ({sign}{quote['day_change_pct']:.2f}%)")
                    if quote.get('bid') is not None and quote.get('ask') is not None:
                        print(f"  Bid:         ${quote['bid']:.2f}  x {quote.get('bid_size', '?')}")
                        print(f"  Ask:         ${quote['ask']:.2f}  x {quote.get('ask_size', '?')}")
                        print(f"  Spread:      ${quote['spread']:.4f}")
                    if quote.get('day_high') is not None:
                        print(f"  Day Range:   ${quote['day_low']:.2f} – ${quote['day_high']:.2f}")
                    if quote.get('prev_close') is not None:
                        print(f"  Prev Close:  ${quote['prev_close']:.2f}")
                    print(f"  Volume:      {int(quote['v']):,}")

                    # Latest 15-minute candle
                    if quote.get('bar_time'):
                        bar_chg = quote['bar_close'] - quote['bar_open']
                        bar_sign = '+' if bar_chg >= 0 else ''
                        print(f"\n  Latest 15m Bar ({quote['bar_time']})")
                        print(f"    O: ${quote['bar_open']:.2f}  H: ${quote['bar_high']:.2f}  "
                              f"L: ${quote['bar_low']:.2f}  C: ${quote['bar_close']:.2f}")
                        print(f"    Change: {bar_sign}${bar_chg:.2f}   Vol: {quote['bar_volume']:,}")

                # Account & position snapshot
                account = self.executor.get_account_info()
                position = self.executor.get_position()

                print(f"\n{'─'*50}")
                print(f"  Account Snapshot")
                print(f"{'─'*50}")
                if account:
                    print(f"  Cash:            ${account['cash']:.2f}")
                    print(f"  Buying Power:    ${account['buying_power']:.2f}")
                    print(f"  Portfolio Value:  ${account['portfolio_value']:.2f}")
                    print(f"  Equity:          ${account['equity']:.2f}")
                if position:
                    print(f"  Position:        {position['qty']} shares @ ${position['avg_entry_price']:.2f}")
                    pl_sign = '+' if position['unrealized_pl'] >= 0 else ''
                    print(f"  Unrealized P&L:  {pl_sign}${position['unrealized_pl']:.2f} ({pl_sign}{position['unrealized_plpc']:.2%})")
                else:
                    print(f"  Position:        None")
                print()
                
                # ── Check stop-loss / take-profit first ──────────────
                position = self.executor.get_position()
                if position and quote:
                    sl_tp_action = self._check_stop_loss_take_profit(quote['c'])
                    if sl_tp_action == 'sell':
                        self.execute_trade('sell', confidence=0.0)
                        print(f"Next check in {check_interval}s...")
                        time.sleep(check_interval)
                        continue

                # Make prediction
                action, confidence = self.decide_trade(features)
                
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
