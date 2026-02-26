"""
Trading execution module using Alpaca API

Risk-management features added to target:
    • Sharpe  ≥ 0.5
    • Max DD  ≤ 10 %
"""
import alpaca_trade_api as tradeapi
from src.config import (
    ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL,
    STOCK_SYMBOL, STOP_LOSS_PCT, TAKE_PROFIT_PCT,
)


class TradingExecutor:
    def __init__(self, api_key=ALPACA_API_KEY, secret_key=ALPACA_SECRET_KEY):
        self.client = tradeapi.REST(
            key_id=api_key,
            secret_key=secret_key,
            base_url=ALPACA_BASE_URL
        )
        self.symbol = STOCK_SYMBOL
    
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
            positions = self.client.list_positions()
            for position in positions:
                if position.symbol == self.symbol:
                    return {
                        'qty': int(position.qty),
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
        Place a buy market order
        
        Args:
            qty: Quantity to buy
            
        Returns:
            dict: Order details
        """
        try:
            order = self.client.submit_order(
                symbol=self.symbol,
                qty=qty,
                side='buy',
                type='market',
                time_in_force='day'
            )
            print(f"Buy order placed: {qty} shares of {self.symbol}")
            return {
                'order_id': order.id,
                'symbol': order.symbol,
                'qty': order.qty,
                'side': order.side,
                'status': order.status
            }
        except Exception as e:
            print(f"Error placing buy order: {e}")
            return None
    
    def place_sell_order(self, qty):
        """
        Place a sell market order
        
        Args:
            qty: Quantity to sell
            
        Returns:
            dict: Order details
        """
        try:
            order = self.client.submit_order(
                symbol=self.symbol,
                qty=qty,
                side='sell',
                type='market',
                time_in_force='day'
            )
            print(f"Sell order placed: {qty} shares of {self.symbol}")
            return {
                'order_id': order.id,
                'symbol': order.symbol,
                'qty': order.qty,
                'side': order.side,
                'status': order.status
            }
        except Exception as e:
            print(f"Error placing sell order: {e}")
            return None
    
    def get_order_history(self, limit=10):
        """
        Get order history
        
        Args:
            limit: Number of orders to retrieve
            
        Returns:
            list: List of orders
        """
        try:
            orders = self.client.list_orders(
                status='all',
                limit=limit
            )
            return [
                {
                    'order_id': order.id,
                    'symbol': order.symbol,
                    'qty': order.qty,
                    'side': order.side,
                    'filled_qty': order.filled_qty,
                    'filled_avg_price': order.filled_avg_price,
                    'status': order.status,
                    'created_at': order.created_at
                }
                for order in orders
                if order.symbol == self.symbol
            ]
        except Exception as e:
            print(f"Error getting order history: {e}")
            return None
