"""
Alpaca order management module

Handles all interactions with the Alpaca brokerage API:
    • Account information
    • Position queries
    • Order placement (buy / sell)
    • Order history
"""
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
from config.settings import (
    ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL,
    STOCK_SYMBOL, STOP_LOSS_PCT, TAKE_PROFIT_PCT,
    is_crypto, alpaca_symbol,
)


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
    
    def get_position(self):
        """
        Get current position in the stock
        
        Returns:
            dict: Position details or None if no position
        """
        try:
            positions = self.client.get_all_positions()
            for position in positions:
                if position.symbol == self.symbol:
                    return {
                        'qty': int(float(position.qty)),
                        'avg_entry_price': float(position.avg_entry_price),
                        'current_price': float(position.current_price),
                        'unrealized_pl': float(position.unrealized_pl),
                        'unrealized_plpc': float(position.unrealized_plpc)
                    }
            return None
        except Exception as e:
            print(f"Error getting position: {e}")
            return None
    
    def place_buy_order(self, qty):
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
    
    def place_sell_order(self, qty):
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
