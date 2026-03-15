"""
Trade logging utility for recording trading activity and performance.
"""
from datetime import datetime, timezone

from utils.db_connector import init_database, insert_trade_log


class TradeLogger:
    """Log trades to PostgreSQL for audit and analysis."""

    def __init__(self):
        init_database()

    def log(self, action, symbol, qty, price, confidence=0.0,
            investment=0.0, note=''):
        """
        Append a single trade record.

        Args:
            action: 'BUY', 'SELL', 'SELL (SL)', 'SELL (TP)', etc.
            symbol: Ticker symbol
            qty: Number of shares
            price: Execution price
            confidence: Model confidence (0.0 – 1.0)
            investment: Dollar investment amount
            note: Free-text note
        """
        insert_trade_log(
            event_time=datetime.now(timezone.utc),
            symbol=symbol,
            action=action,
            qty=float(qty),
            price=round(float(price), 4),
            confidence=round(float(confidence), 4),
            investment=round(float(investment), 2),
            venue="alpaca",
            mode="live",
            notes=note,
        )
