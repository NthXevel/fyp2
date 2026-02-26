"""
Data fetching module for retrieving stock data from Yahoo Finance
"""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from src.config import STOCK_SYMBOL, DATA_INTERVAL, DATA_DAYS


class DataFetcher:
    def __init__(self, symbol=STOCK_SYMBOL, interval=DATA_INTERVAL):
        self.symbol = symbol
        self.interval = interval
        self.ticker = yf.Ticker(symbol)
    
    def get_historical_data(self, days=DATA_DAYS, interval=None):
        """
        Fetch historical stock data for the past N days
        
        Args:
            days: Number of days of historical data to fetch
                  (Yahoo Finance max ~60 days for 15m interval)
            interval: Data interval (default from config: '15m')
            
        Returns:
            pd.DataFrame: DataFrame with OHLCV data
        """
        if interval is None:
            interval = self.interval
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Fetch data from Yahoo Finance at configured interval
            df = yf.download(
                self.symbol,
                start=start_date,
                end=end_date,
                interval=interval,
                progress=False
            )
            
            # Handle MultiIndex columns (when downloading single ticker)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # Rename columns to match expected format
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume',
                'Adj Close': 'adj_close'
            })
            
            # Keep only required columns (handle if Adj Close doesn't exist)
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            available_cols = [col for col in required_cols if col in df.columns]
            df = df[available_cols]
            
            # Reset and clean the index
            df.index.name = 'date'
            
            return df
                
        except Exception as e:
            print(f"Exception occurred while fetching data: {e}")
            return None
    
    def get_realtime_quote(self):
        """
        Get real-time quote including bid/ask and the latest 15m candle.
        
        Returns:
            dict: Current stock quote data with bid/ask and 15m bar info
        """
        try:
            # Fetch 1-minute data for the most up-to-date price
            data_1m = self.ticker.history(period='1d', interval='1m')
            # Fetch latest 15m candle
            data_15m = self.ticker.history(period='5d', interval='15m')
            info = self.ticker.info

            if data_1m.empty:
                return None

            latest = data_1m.iloc[-1]
            bid = info.get('bid', None)
            ask = info.get('ask', None)
            bid_size = info.get('bidSize', None)
            ask_size = info.get('askSize', None)
            spread = round(ask - bid, 4) if bid and ask else None
            prev_close = info.get('previousClose', None)
            current = latest['Close']
            day_change = round(current - prev_close, 2) if prev_close else None
            day_change_pct = round((current - prev_close) / prev_close * 100, 2) if prev_close else None

            result = {
                'c': current,
                'h': latest['High'],
                'l': latest['Low'],
                'o': latest['Open'],
                'v': latest['Volume'],
                't': int(data_1m.index[-1].timestamp()),
                'bid': bid,
                'bid_size': bid_size,
                'ask': ask,
                'ask_size': ask_size,
                'spread': spread,
                'prev_close': prev_close,
                'day_change': day_change,
                'day_change_pct': day_change_pct,
                'day_high': info.get('dayHigh', None),
                'day_low': info.get('dayLow', None),
                'fifty_day_avg': info.get('fiftyDayAverage', None),
                'two_hundred_day_avg': info.get('twoHundredDayAverage', None),
            }

            # Attach latest 15m candle info
            if not data_15m.empty:
                bar = data_15m.iloc[-1]
                result['bar_time'] = str(data_15m.index[-1])
                result['bar_open'] = bar['Open']
                result['bar_high'] = bar['High']
                result['bar_low'] = bar['Low']
                result['bar_close'] = bar['Close']
                result['bar_volume'] = int(bar['Volume'])

            return result
        except Exception as e:
            print(f"Exception occurred while fetching quote: {e}")
            return None
    
    def get_company_info(self):
        """
        Get company information
        
        Returns:
            dict: Company info data
        """
        try:
            info = self.ticker.info
            return {
                'name': info.get('longName', ''),
                'industry': info.get('industry', ''),
                'sector': info.get('sector', ''),
                'marketCap': info.get('marketCap', 0),
                'description': info.get('longBusinessSummary', '')
            }
        except Exception as e:
            print(f"Exception occurred while fetching company info: {e}")
            return None
