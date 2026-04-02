"""
Backtesting engine for evaluating trading strategies on historical data.

Improvements over the previous version:
- Next-bar execution at bar open (instead of same-bar close fill)
- Configurable spread + slippage
- Intrabar stop-loss / take-profit using high/low when available
- Win rate based on per-trade P&L
- Sharpe annualised with the correct bar frequency
"""

from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
import pandas as pd

from config.settings import (
    STOP_LOSS_PCT,
    TAKE_PROFIT_PCT,
    TARGET_SHARPE,
    TARGET_MAX_DRAWDOWN,
    TARGET_ACCURACY,
    MIN_PCT_ALLOCATION,
    MAX_PCT_ALLOCATION,
)
from strategies.feature_engineering import FORWARD_BARS
from utils.db_connector import init_database, insert_trade_log, upsert_backtest_metric

# Approximate number of 15-minute bars in a US-equity trading year (6.5h/day * 4 bars/hour * 252)
BARS_PER_YEAR = 26 * 252


class BacktestEngine:
    """Long-only backtest engine with next-bar execution, confidence sizing, and SL/TP."""

    def __init__(
        self,
        initial_capital=100_000,
        transaction_cost=0.001,
        min_pct=MIN_PCT_ALLOCATION,
        max_pct=MAX_PCT_ALLOCATION,
        stop_loss=STOP_LOSS_PCT,
        take_profit=TAKE_PROFIT_PCT,
        confidence_threshold=0.5,
        slippage_bps=5.0,
        spread_bps=2.0,
        periods_per_year=BARS_PER_YEAR,
    ):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.min_pct = min_pct
        self.max_pct = max_pct
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.confidence_threshold = confidence_threshold
        self.slippage_bps = slippage_bps
        self.spread_bps = spread_bps
        self.periods_per_year = periods_per_year
        self._last_run_id = None

        init_database()

    @property
    def _execution_impact(self) -> float:
        """One-sided execution penalty: half-spread + slippage."""
        return (self.spread_bps / 20000.0) + (self.slippage_bps / 10000.0)

    def _buy_exec_price(self, row: pd.Series) -> float:
        base = row["open"] if "open" in row and pd.notna(row["open"]) else row["close"]
        return float(base) * (1.0 + self._execution_impact)

    def _sell_exec_price(self, row: pd.Series) -> float:
        base = row["open"] if "open" in row and pd.notna(row["open"]) else row["close"]
        return float(base) * (1.0 - self._execution_impact)

    def _sell_exec_price_from_level(self, level: float) -> float:
        """Used for SL/TP fills when a trigger level is hit intrabar."""
        return float(level) * (1.0 - self._execution_impact)

    def _mark_price(self, row: pd.Series) -> float:
        return float(row["close"])

    def _position_dollars(self, capital: float, prob_up: float) -> float:
        scale = (prob_up - self.confidence_threshold) / (1.0 - self.confidence_threshold)
        scale = max(0.0, min(1.0, scale))

        current_min_dollars = capital * self.min_pct
        current_max_dollars = capital * self.max_pct

        investment = current_min_dollars + scale * (current_max_dollars - current_min_dollars)
        return min(investment, capital)

    def _close_position(
        self,
        date,
        action: str,
        exec_price: float,
        shares: float,
        capital: float,
        confidence: float,
        trade_log: list,
        open_trade: dict,
    ) -> float:
        revenue = shares * exec_price * (1.0 - self.transaction_cost)
        capital += revenue

        pnl = revenue - open_trade["entry_cost"]
        trade_return = pnl / open_trade["entry_cost"] if open_trade["entry_cost"] else 0.0

        trade_log.append(
            {
                "date": date,
                "action": action,
                "price": exec_price,
                "shares": shares,
                "capital": capital,
                "confidence": confidence,
                "investment": 0.0,
                "trade_id": open_trade["trade_id"],
                "pnl": pnl,
                "trade_return": trade_return,
            }
        )
        return capital

    def run(
        self,
        test_df,
        predictions,
        probabilities,
        symbol="BTC/USD",
        timeframe="15m",
        run_id=None,
        persist=True,
    ):
        """
        Execute a long-only backtest.

        Key assumptions:
        - Signals are generated on bar t close
        - Orders are executed on bar t+1 open
        - Intrabar stop-loss / take-profit uses high/low if available
        - Positions are held for at least FORWARD_BARS completed bars
        """
        required_cols = {"close"}
        missing = required_cols - set(test_df.columns)
        if missing:
            raise ValueError(f"test_df missing required columns: {missing}")

        if len(test_df) != len(predictions) or len(test_df) != len(probabilities):
            raise ValueError("test_df, predictions, and probabilities must have the same length")

        capital = self.initial_capital
        shares = 0.0
        entry_price = 0.0
        bars_held = 0
        min_hold = FORWARD_BARS

        portfolio_values = []
        trade_log = []

        pending_order = None
        open_trade = None

        for i, (date, row) in enumerate(test_df.iterrows()):
            pred = int(predictions[i])
            prob_up = float(probabilities[i, 1])

            # 1) Execute pending order from previous bar at THIS bar open
            if pending_order is not None:
                side = pending_order["side"]
                reason = pending_order["reason"]
                order_confidence = pending_order["confidence"]

                if side == "buy" and shares == 0:
                    exec_price = self._buy_exec_price(row)
                    investment = self._position_dollars(capital, order_confidence)

                    new_shares = round(investment / (exec_price * (1.0 + self.transaction_cost)), 5)
                    cost = new_shares * exec_price * (1.0 + self.transaction_cost)

                    if new_shares > 0 and cost <= capital:
                        capital -= cost
                        shares = new_shares
                        entry_price = exec_price
                        bars_held = 0

                        open_trade = {
                            "trade_id": uuid4().hex[:12],
                            "entry_date": date,
                            "entry_price": exec_price,
                            "entry_cost": cost,
                            "shares": new_shares,
                        }

                        trade_log.append(
                            {
                                "date": date,
                                "action": reason,
                                "price": exec_price,
                                "shares": shares,
                                "capital": capital,
                                "confidence": order_confidence,
                                "investment": cost,
                                "trade_id": open_trade["trade_id"],
                                "pnl": np.nan,
                                "trade_return": np.nan,
                            }
                        )

                elif side == "sell" and shares > 0 and open_trade is not None:
                    exec_price = self._sell_exec_price(row)
                    capital = self._close_position(
                        date=date,
                        action=reason,
                        exec_price=exec_price,
                        shares=shares,
                        capital=capital,
                        confidence=order_confidence,
                        trade_log=trade_log,
                        open_trade=open_trade,
                    )
                    shares = 0.0
                    entry_price = 0.0
                    bars_held = 0
                    open_trade = None

                pending_order = None

            # 2) Intrabar SL/TP check after any open-fill
            if shares > 0 and open_trade is not None:
                low_price = float(row["low"]) if "low" in row and pd.notna(row["low"]) else float(row["close"])
                high_price = float(row["high"]) if "high" in row and pd.notna(row["high"]) else float(row["close"])

                stop_level = entry_price * (1.0 - self.stop_loss)
                take_level = entry_price * (1.0 + self.take_profit)

                # Conservative assumption: if both are touched in the same bar, stop-loss happens first
                if low_price <= stop_level:
                    exec_price = self._sell_exec_price_from_level(stop_level)
                    capital = self._close_position(
                        date=date,
                        action="SELL (SL)",
                        exec_price=exec_price,
                        shares=shares,
                        capital=capital,
                        confidence=prob_up,
                        trade_log=trade_log,
                        open_trade=open_trade,
                    )
                    shares = 0.0
                    entry_price = 0.0
                    bars_held = 0
                    open_trade = None

                elif high_price >= take_level:
                    exec_price = self._sell_exec_price_from_level(take_level)
                    capital = self._close_position(
                        date=date,
                        action="SELL (TP)",
                        exec_price=exec_price,
                        shares=shares,
                        capital=capital,
                        confidence=prob_up,
                        trade_log=trade_log,
                        open_trade=open_trade,
                    )
                    shares = 0.0
                    entry_price = 0.0
                    bars_held = 0
                    open_trade = None

            # 3) Mark to market at close
            mark_price = self._mark_price(row)
            portfolio_value = capital + shares * mark_price
            portfolio_values.append(
                {
                    "date": date,
                    "PortfolioValue": portfolio_value,
                    "Capital": capital,
                    "Shares": shares,
                }
            )

            # 4) End-of-bar holding count
            if shares > 0:
                bars_held += 1

            # 5) Generate signal at close for NEXT bar open execution
            if i < len(test_df) - 1:
                if shares == 0 and pred == 1 and prob_up >= self.confidence_threshold:
                    pending_order = {
                        "side": "buy",
                        "reason": "BUY",
                        "confidence": prob_up,
                    }

                elif shares > 0 and bars_held >= min_hold and pred == 0:
                    pending_order = {
                        "side": "sell",
                        "reason": "SELL",
                        "confidence": prob_up,
                    }

        results = pd.DataFrame(portfolio_values).set_index("date")
        trade_log_df = pd.DataFrame(trade_log)

        # 6) Force-close any remaining open position at final close
        if shares > 0 and open_trade is not None and not results.empty:
            last_date = results.index[-1]
            last_close = float(test_df.iloc[-1]["close"])
            final_exec_price = self._sell_exec_price_from_level(last_close)

            capital = self._close_position(
                date=last_date,
                action="SELL (EOD)",
                exec_price=final_exec_price,
                shares=shares,
                capital=capital,
                confidence=0.0,
                trade_log=trade_log,
                open_trade=open_trade,
            )
            shares = 0.0
            entry_price = 0.0
            bars_held = 0
            open_trade = None

            trade_log_df = pd.DataFrame(trade_log)
            results.iloc[-1, results.columns.get_loc("PortfolioValue")] = capital
            results.iloc[-1, results.columns.get_loc("Capital")] = capital
            results.iloc[-1, results.columns.get_loc("Shares")] = 0.0

        run_key = run_id or f"backtest-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{uuid4().hex[:8]}"
        self._last_run_id = run_key

        if persist and not trade_log_df.empty:
            self._persist_trade_log(trade_log_df, symbol=symbol, run_id=run_key)

        return results, trade_log_df

    @staticmethod
    def _persist_trade_log(trade_log: pd.DataFrame, symbol: str, run_id: str) -> None:
        """Write backtest trade events to trade_logs table."""
        for _, row in trade_log.iterrows():
            event_time = row["date"]
            if isinstance(event_time, pd.Timestamp):
                event_time = event_time.to_pydatetime()
            if event_time.tzinfo is None:
                event_time = event_time.replace(tzinfo=timezone.utc)

            insert_trade_log(
                event_time=event_time,
                symbol=symbol,
                action=str(row["action"]),
                qty=float(row["shares"]),
                price=float(row["price"]),
                confidence=float(row.get("confidence", 0.0)),
                investment=float(row.get("investment", 0.0)),
                capital=float(row.get("capital", 0.0)),
                venue="backtest",
                mode="backtest",
                run_id=run_id,
            )

    @staticmethod
    def calc_sharpe(returns, periods=252):
        """Annualised Sharpe ratio from bar-level returns."""
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        return float(np.sqrt(periods) * returns.mean() / returns.std())

    @staticmethod
    def calc_max_drawdown(portfolio_values):
        """Maximum drawdown from a portfolio value series."""
        cum_max = portfolio_values.cummax()
        drawdown = (portfolio_values - cum_max) / cum_max
        return float(drawdown.min())

    def summary(
        self,
        results,
        trade_log,
        test_accuracy,
        symbol="BTC/USD",
        timeframe="15m",
        run_id=None,
        persist=True,
    ):
        """Compute and print a performance summary."""
        if results.empty:
            raise ValueError("results is empty; cannot compute summary")

        bar_returns = results["PortfolioValue"].pct_change().dropna()
        final_capital = float(results["PortfolioValue"].iloc[-1])
        cum_return = (final_capital - self.initial_capital) / self.initial_capital

        sharpe = self.calc_sharpe(bar_returns, periods=self.periods_per_year)
        max_dd = self.calc_max_drawdown(results["PortfolioValue"])

        num_trades = len(trade_log)
        num_buys = int((trade_log["action"] == "BUY").sum()) if len(trade_log) else 0
        sell_mask = trade_log["action"].astype(str).str.startswith("SELL") if len(trade_log) else pd.Series(dtype=bool)
        num_sells = int(sell_mask.sum()) if len(trade_log) else 0

        win_rate = 0.0
        if len(trade_log) and "pnl" in trade_log.columns and num_sells > 0:
            closed_trades = trade_log[sell_mask].copy()
            win_rate = float((closed_trades["pnl"] > 0).mean())

        def check(val, target, higher_better=True):
            return "[Y]" if (val >= target if higher_better else val <= target) else "[X]"

        print(f"Initial capital: ${self.initial_capital:,.2f}")
        print(f"Final capital:   ${final_capital:,.2f}")
        print(f"Cumulative return: {cum_return:.2%}")
        print()
        print(f" Accuracy: {test_accuracy:.4f} {check(test_accuracy, TARGET_ACCURACY)} target >= {TARGET_ACCURACY}")
        print(f" Sharpe:   {sharpe:.3f} {check(sharpe, TARGET_SHARPE)} target >= {TARGET_SHARPE}")
        print(f" Max DD:   {max_dd:.2%} {check(abs(max_dd), TARGET_MAX_DRAWDOWN, higher_better=False)} target <= {TARGET_MAX_DRAWDOWN:.0%}")
        print(f" Win rate: {win_rate:.2%}")
        print(f" Trades:   {num_trades} (buys: {num_buys}, sells: {num_sells})")

        metrics = {
            "initial_capital": self.initial_capital,
            "final_capital": round(final_capital, 2),
            "cumulative_return": round(cum_return, 4),
            "sharpe_ratio": round(sharpe, 3),
            "max_drawdown": round(max_dd, 4),
            "win_rate": round(win_rate, 4),
            "num_trades": num_trades,
            "test_accuracy": round(test_accuracy, 4),
        }

        if persist:
            run_key = run_id or self._last_run_id or f"backtest-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{uuid4().hex[:8]}"
            upsert_backtest_metric(
                run_id=run_key,
                symbol=symbol,
                timeframe=timeframe,
                initial_capital=float(self.initial_capital),
                final_capital=float(metrics["final_capital"]),
                cumulative_return=float(metrics["cumulative_return"]),
                sharpe_ratio=float(metrics["sharpe_ratio"]),
                max_drawdown=float(metrics["max_drawdown"]),
                win_rate=float(metrics["win_rate"]),
                num_trades=int(metrics["num_trades"]),
                test_accuracy=float(metrics["test_accuracy"]),
                metadata={
                    "transaction_cost": self.transaction_cost,
                    "confidence_threshold": self.confidence_threshold,
                    "stop_loss": self.stop_loss,
                    "take_profit": self.take_profit,
                    "slippage_bps": self.slippage_bps,
                    "spread_bps": self.spread_bps,
                    "periods_per_year": self.periods_per_year,
                },
            )

        return metrics