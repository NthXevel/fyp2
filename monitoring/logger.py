"""
Trade logging utility for recording trading activity and performance.
"""
import os
import csv
from datetime import datetime


class TradeLogger:
    """Log trades to a CSV file for audit and analysis."""

    def __init__(self, log_dir='reports', filename='live_trade_log.csv'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.filepath = os.path.join(log_dir, filename)
        self._ensure_header()

    def _ensure_header(self):
        """Create the CSV file with a header row if it does not exist."""
        if not os.path.exists(self.filepath):
            with open(self.filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'action', 'symbol', 'qty',
                    'price', 'confidence', 'investment', 'note',
                ])

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
        with open(self.filepath, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                action,
                symbol,
                qty,
                round(price, 4),
                round(confidence, 4),
                round(investment, 2),
                note,
            ])
