"""
Signal generation and position sizing module

Handles:
    • Trade signal generation (buy / sell / hold) based on model predictions
    • Confidence-based position sizing
    • Per-order stop-loss and take-profit risk management
"""
from config.settings import (
    CONFIDENCE_THRESHOLD, MIN_PCT_ALLOCATION, MAX_PCT_ALLOCATION,
    STOP_LOSS_PCT, TAKE_PROFIT_PCT,
)


class SignalGenerator:
    """Generate trading signals and manage position sizing / risk."""

    def __init__(self, confidence_threshold=CONFIDENCE_THRESHOLD,
                 min_pct=MIN_PCT_ALLOCATION,
                 max_pct=MAX_PCT_ALLOCATION,
                 stop_loss_pct=STOP_LOSS_PCT,
                 take_profit_pct=TAKE_PROFIT_PCT):
        self.confidence_threshold = confidence_threshold
        self.min_pct = min_pct
        self.max_pct = max_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

    def decide_trade(self, probabilities):
        """
        Decide whether to buy, sell, or hold based on model probabilities.

        Args:
            probabilities: Model predict_proba output for a single sample
                           [[prob_down, prob_up]]

        Returns:
            tuple: (action: str, confidence: float)
                   action is 'buy', 'sell', or 'hold'
        """
        prob_down = probabilities[0][0]
        prob_up = probabilities[0][1]

        if prob_up > self.confidence_threshold:
            return 'buy', prob_up
        elif prob_down > self.confidence_threshold:
            return 'sell', prob_down
        else:
            return 'hold', max(prob_up, prob_down)

    def calculate_position_size(self, confidence, current_price, equity, fractional=False):
        """
        Calculate the quantity to buy based on confidence and account equity.

        The investment amount is a percentage of equity, scaled linearly
        between MIN_PCT_ALLOCATION and MAX_PCT_ALLOCATION based on how
        far above the confidence threshold the model's prediction is.

        Args:
            confidence: Model prediction probability (0.0 – 1.0)
            current_price: Current price
            equity: Current account equity
            fractional: If True, return fractional qty (for crypto).
                        Otherwise return integer qty (for stocks).

        Returns:
            tuple: (qty: int|float, investment: float)
        """
        scale = (confidence - self.confidence_threshold) / (1.0 - self.confidence_threshold)
        scale = max(0.0, min(1.0, scale))
        # Dynamic dollar limits based on current equity
        current_min_dollars = equity * self.min_pct
        current_max_dollars = equity * self.max_pct
        investment = current_min_dollars + scale * (current_max_dollars - current_min_dollars)
        if fractional:
            qty = round(investment / current_price, 6)  # up to 6 decimal places
        else:
            qty = int(investment / current_price)
        return qty, investment

    def check_stop_loss_take_profit(self, entry_price, current_price):
        """
        Check whether a stop-loss or take-profit condition is triggered.

        SL/TP are evaluated on the *order* entry price, not total portfolio.

        Args:
            entry_price: Price at which the position was opened
            current_price: Current market price

        Returns:
            str or None: 'sell' if SL or TP hit, else None
        """
        if entry_price is None:
            return None

        change = (current_price - entry_price) / entry_price

        if change <= -self.stop_loss_pct:
            print(f"STOP-LOSS triggered ({change:.2%} vs -{self.stop_loss_pct:.1%}) [per-order]")
            return 'sell'
        if change >= self.take_profit_pct:
            print(f"TAKE-PROFIT triggered ({change:.2%} vs +{self.take_profit_pct:.1%}) [per-order]")
            return 'sell'

        return None
