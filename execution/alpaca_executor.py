"""
Alpaca order management module

Handles all interactions with the Alpaca brokerage API:
    • Account information
    • Position queries
    • Order placement (buy / sell)
    • Order history
"""
from datetime import datetime, timezone

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
from config.settings import (
    ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL,
    STOCK_SYMBOL, STOP_LOSS_PCT, TAKE_PROFIT_PCT,
    is_crypto, alpaca_symbol,
)
from utils.db_connector import init_database, insert_trade_log


class TradingExecutor:
    def __init__(self, api_key=ALPACA_API_KEY, secret_key=ALPACA_SECRET_KEY):
        if not api_key or not secret_key:
            raise ValueError(
                "Alpaca credentials not found. "
                "Set ALPACA_API_KEY and ALPACA_SECRET_KEY in your "
                ".env file, .streamlit/secrets.toml, or Streamlit Cloud Secrets."
            )
        paper = 'paper' in ALPACA_BASE_URL.lower()
        self.client = TradingClient(
            api_key=api_key,
            secret_key=secret_key,
            paper=paper,
        )
        self.symbol = alpaca_symbol(STOCK_SYMBOL)
        self._is_crypto = is_crypto(STOCK_SYMBOL)
        try:
            init_database()
        except Exception as exc:
            print(f"Warning: database init failed in TradingExecutor: {exc}")

    def _log_trade_event(
        self,
        *,
        action: str,
        symbol: str,
        qty: float,
        price: float,
        confidence: float | None = None,
        investment: float | None = None,
        capital: float | None = None,
        order_id: str | None = None,
        status: str | None = None,
        note: str | None = None,
        run_id: str | None = None,
    ) -> None:
        try:
            insert_trade_log(
                event_time=datetime.now(timezone.utc),
                symbol=symbol,
                action=action,
                qty=float(qty),
                price=float(price),
                confidence=confidence,
                investment=investment,
                capital=capital,
                order_id=order_id,
                status=status,
                venue="alpaca",
                mode="live",
                run_id=run_id,
                notes=note,
            )
        except Exception as exc:
            print(f"Warning: failed to log trade to database: {exc}")
    
    def get_account_info(self):
        """
        Get account information
        
        Returns:
            dict: Account details
        """
        try:
            account = self.client.get_account()
            return {
                'buying_power': float(account.buying_power),
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value),
                'equity': float(account.equity)
            }
        except Exception as e:
            print(f"Error getting account info: {e}")
            return None
    
    @staticmethod
    def _norm_symbol(s: str) -> str:
        """Normalise symbol for comparison (strip / and - , upper-case)."""
        return s.replace('/', '').replace('-', '').upper()

    def get_position(self):
        """
        Get current position in the stock
        
        Returns:
            dict: Position details or None if no position
        """
        try:
            positions = self.client.get_all_positions()
            target = self._norm_symbol(self.symbol)
            for position in positions:
                if self._norm_symbol(position.symbol) == target:
                    return {
                        'qty': float(position.qty),
                        'avg_entry_price': float(position.avg_entry_price),
                        'current_price': float(position.current_price),
                        'unrealized_pl': float(position.unrealized_pl),
                        'unrealized_plpc': float(position.unrealized_plpc)
                    }
            return None
        except Exception as e:
            print(f"Error getting position: {e}")
            return None
    
    def place_buy_order(self, qty, confidence=None, investment=None, capital=None, note="", run_id=None):
        """
        Place a buy market order.
        Crypto orders use 'gtc' time-in-force and support fractional qty.
        
        Args:
            qty: Quantity to buy (int for stocks, float for crypto)
            
        Returns:
            dict: Order details
        """
        try:
            tif = TimeInForce.GTC if self._is_crypto else TimeInForce.DAY
            order_data = MarketOrderRequest(
                symbol=self.symbol,
                qty=qty,
                side=OrderSide.BUY,
                time_in_force=tif,
            )
            order = self.client.submit_order(order_data)
            print(f"Buy order placed: {qty} shares of {self.symbol}")
            fill_price = float(order.filled_avg_price) if order.filled_avg_price else None
            self._log_trade_event(
                action="BUY",
                symbol=order.symbol,
                qty=float(order.qty),
                price=fill_price if fill_price is not None else float(investment / float(qty)) if investment and qty else 0.0,
                confidence=confidence,
                investment=investment,
                capital=capital,
                order_id=str(order.id),
                status=str(order.status),
                note=note,
                run_id=run_id,
            )
            return {
                'order_id': str(order.id),
                'symbol': order.symbol,
                'qty': str(order.qty),
                'side': str(order.side),
                'status': str(order.status)
            }
        except Exception as e:
            print(f"Error placing buy order: {e}")
            return None
    
    def place_sell_order(self, qty, confidence=None, capital=None, note="", run_id=None):
        """
        Place a sell market order.
        Crypto orders use 'gtc' time-in-force and support fractional qty.
        
        Args:
            qty: Quantity to sell (int for stocks, float for crypto)
            
        Returns:
            dict: Order details
        """
        try:
            tif = TimeInForce.GTC if self._is_crypto else TimeInForce.DAY
            order_data = MarketOrderRequest(
                symbol=self.symbol,
                qty=qty,
                side=OrderSide.SELL,
                time_in_force=tif,
            )
            order = self.client.submit_order(order_data)
            print(f"Sell order placed: {qty} shares of {self.symbol}")
            fill_price = float(order.filled_avg_price) if order.filled_avg_price else 0.0
            self._log_trade_event(
                action="SELL",
                symbol=order.symbol,
                qty=float(order.qty),
                price=fill_price,
                confidence=confidence,
                capital=capital,
                order_id=str(order.id),
                status=str(order.status),
                note=note,
                run_id=run_id,
            )
            return {
                'order_id': str(order.id),
                'symbol': order.symbol,
                'qty': str(order.qty),
                'side': str(order.side),
                'status': str(order.status)
            }
        except Exception as e:
            print(f"Error placing sell order: {e}")
            return None
    
    def get_order_history(self, limit=10, symbol_filter=None):
        """
        Get order history
        
        Args:
            limit: Number of orders to retrieve
            symbol_filter: Optional symbol to filter by (None = all symbols)
            
        Returns:
            list: List of orders
        """
        try:
            request = GetOrdersRequest(
                status=QueryOrderStatus.ALL,
                limit=limit,
            )
            orders = self.client.get_orders(request)
            results = []
            for order in orders:
                # Normalise both sides so BTC/USD, BTCUSD, btc/usd all match
                if symbol_filter:
                    norm = lambda s: s.replace('/', '').replace('-', '').upper()
                    if norm(order.symbol) != norm(symbol_filter):
                        continue
                results.append({
                    'order_id': str(order.id),
                    'symbol': order.symbol,
                    'qty': str(order.qty),
                    'side': str(order.side),
                    'filled_qty': str(order.filled_qty),
                    'filled_avg_price': str(order.filled_avg_price) if order.filled_avg_price else None,
                    'status': str(order.status),
                    'created_at': str(order.created_at),
                })
            return results
        except Exception as e:
            print(f"Error getting order history: {e}")
            raise

    # ── Methods used by the Streamlit dashboard ───────────────────────

    def get_all_positions(self):
        """
        Return a dict keyed by symbol with position details for every
        open position in the account.
        """
        try:
            positions = self.client.get_all_positions()
            return {
                p.symbol: {
                    'qty': float(p.qty),
                    'avg_entry_price': float(p.avg_entry_price),
                    'current_price': float(p.current_price),
                    'market_value': float(p.market_value),
                    'unrealized_pl': float(p.unrealized_pl),
                    'unrealized_plpc': float(p.unrealized_plpc),
                }
                for p in positions
            }
        except Exception as e:
            print(f"Error getting all positions: {e}")
            raise

    def get_open_orders(self):
        """Return a list of open Order objects."""
        try:
            request = GetOrdersRequest(status=QueryOrderStatus.OPEN)
            return self.client.get_orders(request)
        except Exception as e:
            print(f"Error getting open orders: {e}")
            raise

    def cancel_all_orders(self):
        """Cancel every open order."""
        try:
            self.client.cancel_orders()
        except Exception as e:
            print(f"Error cancelling orders: {e}")

    def close_all_positions(self):
        """Close every open position."""
        try:
            self.client.close_all_positions(cancel_orders=True)
        except Exception as e:
            print(f"Error closing positions: {e}")
