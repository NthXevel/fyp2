"""
Backtesting engine for evaluating trading strategies on historical data

Supports:
    • Confidence-based position sizing
    • Per-order stop-loss and take-profit
    • Minimum hold period (matches FORWARD_BARS target horizon)
    • Performance metrics: Sharpe ratio, max drawdown, win rate
"""
import numpy as np
import pandas as pd
from config.settings import (
    STOP_LOSS_PCT, TAKE_PROFIT_PCT,
    TARGET_SHARPE, TARGET_MAX_DRAWDOWN, TARGET_ACCURACY,
)
from strategies.feature_engineering import FORWARD_BARS

# Approximate number of 15-minute bars in a trading year
BARS_PER_YEAR = 26 * 252


class BacktestEngine:
    """Long-only backtest engine with confidence-based sizing and SL/TP."""

    def __init__(self, initial_capital=100_000, transaction_cost=0.001,
                 min_investment=200, max_investment=2000,
                 stop_loss=STOP_LOSS_PCT, take_profit=TAKE_PROFIT_PCT,
                 confidence_threshold=0.55):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.min_investment = min_investment
        self.max_investment = max_investment
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.confidence_threshold = confidence_threshold

    def run(self, test_df, predictions, probabilities):
        """
        Execute a long-only backtest.

        Positions are held for at least FORWARD_BARS bars to match the
        target horizon.  Stop-loss / take-profit still override.

        Args:
            test_df: DataFrame with at least a 'close' column (indexed by datetime)
            predictions: Array of 0/1 predictions (same length as test_df)
            probabilities: Array of shape (N, 2) with [prob_down, prob_up]

        Returns:
            tuple: (results_df, trade_log_df)
        """
        capital = self.initial_capital
        shares = 0
        entry_price = 0.0
        bars_held = 0
        min_hold = FORWARD_BARS          # must hold at least this many bars
        portfolio_values = []
        trade_log = []

        for i, (date, row) in enumerate(test_df.iterrows()):
            price = row['close']
            pred = predictions[i]
            prob_up = probabilities[i, 1]

            # ── Check stop-loss / take-profit while holding ──────────
            if shares > 0:
                bars_held += 1
                pnl_pct = (price - entry_price) / entry_price
                if pnl_pct <= -self.stop_loss:
                    revenue = shares * price * (1 - self.transaction_cost)
                    capital += revenue
                    trade_log.append({
                        'date': date, 'action': 'SELL (SL)', 'price': price,
                        'shares': shares, 'capital': capital,
                        'confidence': prob_up, 'investment': 0,
                    })
                    shares = 0
                    entry_price = 0.0
                    bars_held = 0
                elif pnl_pct >= self.take_profit:
                    revenue = shares * price * (1 - self.transaction_cost)
                    capital += revenue
                    trade_log.append({
                        'date': date, 'action': 'SELL (TP)', 'price': price,
                        'shares': shares, 'capital': capital,
                        'confidence': prob_up, 'investment': 0,
                    })
                    shares = 0
                    entry_price = 0.0
                    bars_held = 0

            # ── Buy signal — high confidence only ────────────────────
            if pred == 1 and shares == 0 and prob_up >= self.confidence_threshold:
                scale = (prob_up - self.confidence_threshold) / (1.0 - self.confidence_threshold)
                scale = max(0.0, min(1.0, scale))
                investment = self.min_investment + scale * (self.max_investment - self.min_investment)
                investment = min(investment, capital)
                shares = int(investment / (price * (1 + self.transaction_cost)))
                cost = shares * price * (1 + self.transaction_cost)
                capital -= cost
                entry_price = price
                bars_held = 0
                trade_log.append({
                    'date': date, 'action': 'BUY', 'price': price,
                    'shares': shares, 'capital': capital,
                    'confidence': prob_up, 'investment': investment,
                })

            # ── Sell signal (model says down AND min hold met) ───────
            elif pred == 0 and shares > 0 and bars_held >= min_hold:
                revenue = shares * price * (1 - self.transaction_cost)
                capital += revenue
                trade_log.append({
                    'date': date, 'action': 'SELL', 'price': price,
                    'shares': shares, 'capital': capital,
                    'confidence': prob_up, 'investment': 0,
                })
                shares = 0
                entry_price = 0.0
                bars_held = 0

            portfolio_value = capital + shares * price
            portfolio_values.append({
                'date': date, 'PortfolioValue': portfolio_value,
                'Capital': capital, 'Shares': shares,
            })

        results = pd.DataFrame(portfolio_values).set_index('date')
        trade_log_df = pd.DataFrame(trade_log)
        return results, trade_log_df

    # ── Performance metrics ──────────────────────────────────────────

    @staticmethod
    def calc_sharpe(returns, periods=252):
        """Annualised Sharpe ratio from bar-level returns (default: daily)."""
        if returns.std() == 0:
            return 0
        return np.sqrt(periods) * returns.mean() / returns.std()

    @staticmethod
    def calc_max_drawdown(portfolio_values):
        """Maximum drawdown from a portfolio value series."""
        cum_max = portfolio_values.cummax()
        drawdown = (portfolio_values - cum_max) / cum_max
        return drawdown.min()

    def summary(self, results, trade_log, test_accuracy):
        """
        Compute and print a performance summary.

        Args:
            results: DataFrame with 'PortfolioValue' column
            trade_log: DataFrame of trades
            test_accuracy: Model test accuracy (float)

        Returns:
            dict: Summary metrics
        """
        bar_returns = results['PortfolioValue'].pct_change().dropna()
        final_capital = results['PortfolioValue'].iloc[-1]
        cum_return = (final_capital - self.initial_capital) / self.initial_capital
        sharpe = self.calc_sharpe(bar_returns)
        max_dd = self.calc_max_drawdown(results['PortfolioValue'])
        num_trades = len(trade_log)
        num_buys = (trade_log['action'] == 'BUY').sum() if len(trade_log) else 0
        num_sells = trade_log['action'].str.startswith('SELL').sum() if len(trade_log) else 0

        win_rate = 0.0
        if num_sells > 0:
            sells = trade_log[trade_log['action'].str.startswith('SELL')].reset_index(drop=True)
            buys = trade_log[trade_log['action'] == 'BUY'].reset_index(drop=True)
            paired = min(len(buys), len(sells))
            wins = sum(sells.loc[j, 'price'] > buys.loc[j, 'price'] for j in range(paired))
            win_rate = wins / paired

        def check(val, target, higher_better=True):
            return '[Y]' if (val >= target if higher_better else val <= target) else '[X]'

        print(f"Initial capital:   ${self.initial_capital:,.2f}")
        print(f"Final capital:     ${final_capital:,.2f}")
        print(f"Cumulative return: {cum_return:.2%}")
        print()
        print(f"  Accuracy:    {test_accuracy:.4f}   {check(test_accuracy, TARGET_ACCURACY)} target >= {TARGET_ACCURACY}")
        print(f"  Sharpe:      {sharpe:.3f}    {check(sharpe, TARGET_SHARPE)} target >= {TARGET_SHARPE}")
        print(f"  Max DD:      {max_dd:.2%}   {check(abs(max_dd), TARGET_MAX_DRAWDOWN, higher_better=False)} target <= {TARGET_MAX_DRAWDOWN:.0%}")
        print(f"  Win rate:    {win_rate:.2%}")
        print(f"  Trades:      {num_trades}  (buys: {num_buys}, sells: {num_sells})")

        return {
            'initial_capital': self.initial_capital,
            'final_capital': round(final_capital, 2),
            'cumulative_return': round(cum_return, 4),
            'sharpe_ratio': round(sharpe, 3),
            'max_drawdown': round(max_dd, 4),
            'win_rate': round(win_rate, 4),
            'num_trades': num_trades,
            'test_accuracy': round(test_accuracy, 4),
        }
