"""
Market data fetching utilities.

This module provides functionality to fetch real market data for options pricing,
including stock prices, options chains, and risk-free rates.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta
import time

from models.volatility import VolatilityCalculator


class MarketDataFetcher:
    """
    Fetch real market data for options pricing
    """
    
    def __init__(self):
        self.cache = {}
        self.cache_timeout = 300  # 5 minutes
    
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data is still valid"""
        if key not in self.cache:
            return False
        
        cache_time, _ = self.cache[key]
        return time.time() - cache_time < self.cache_timeout
    
    def _cache_data(self, key: str, data: any):
        """Cache data with timestamp"""
        self.cache[key] = (time.time(), data)
    
    def get_stock_data(self, symbol: str, 
                      period: str = "1y") -> pd.DataFrame:
        """
        Fetch historical stock data
        
        Args:
            symbol: Stock symbol
            period: Data period ("1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max")
        
        Returns:
            DataFrame with OHLCV data
        """
        cache_key = f"stock_data_{symbol}_{period}"
        
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key][1]
        
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if data.empty:
                raise ValueError(f"No data available for {symbol}")
            
            self._cache_data(cache_key, data)
            return data
        
        except Exception as e:
            raise ValueError(f"Error fetching data for {symbol}: {str(e)}")
    
    def get_current_price(self, symbol: str) -> float:
        """
        Get current stock price
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Current stock price
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Try to get current price from info
            if 'currentPrice' in info and info['currentPrice'] is not None:
                return info['currentPrice']
            
            # Fallback to latest close price
            data = self.get_stock_data(symbol, "1d")
            return data['Close'].iloc[-1]
        
        except Exception as e:
            raise ValueError(f"Error fetching current price for {symbol}: {str(e)}")
    
    def get_options_chain(self, symbol: str, 
                         expiration_date: Optional[str] = None) -> Dict:
        """
        Fetch options chain data
        
        Args:
            symbol: Stock symbol
            expiration_date: Specific expiration date (optional)
        
        Returns:
            Dictionary with calls and puts data
        """
        cache_key = f"options_chain_{symbol}_{expiration_date or 'nearest'}"
        
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key][1]
        
        try:
            ticker = yf.Ticker(symbol)
            
            if expiration_date:
                options = ticker.option_chain(expiration_date)
            else:
                # Get nearest expiration
                expirations = ticker.options
                if not expirations:
                    raise ValueError(f"No options data available for {symbol}")
                options = ticker.option_chain(expirations[0])
                expiration_date = expirations[0]
            
            result = {
                "calls": options.calls,
                "puts": options.puts,
                "expiration": expiration_date,
                "underlying_price": self.get_current_price(symbol)
            }
            
            self._cache_data(cache_key, result)
            return result
        
        except Exception as e:
            raise ValueError(f"Error fetching options data for {symbol}: {str(e)}")
    
    def get_risk_free_rate(self) -> float:
        """
        Get current risk-free rate (10-year Treasury yield)
        
        Returns:
            Risk-free rate as decimal
        """
        cache_key = "risk_free_rate"
        
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key][1]
        
        try:
            # Fetch 10-year Treasury yield
            treasury = yf.Ticker("^TNX")
            rate = treasury.info.get('regularMarketPrice', 0.03) / 100
            
            # Ensure reasonable bounds
            rate = max(0.0, min(0.2, rate))  # Between 0% and 20%
            
            self._cache_data(cache_key, rate)
            return rate
        
        except Exception as e:
            # Fallback to reasonable default
            default_rate = 0.03
            self._cache_data(cache_key, default_rate)
            return default_rate
    
    def calculate_historical_volatility(self, symbol: str, 
                                      window: int = 252) -> float:
        """
        Calculate historical volatility for a stock
        
        Args:
            symbol: Stock symbol
            window: Number of trading days
        
        Returns:
            Annualized volatility
        """
        cache_key = f"hist_vol_{symbol}_{window}"
        
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key][1]
        
        try:
            data = self.get_stock_data(symbol, "1y")
            volatility = VolatilityCalculator.historical_volatility(
                data['Close'].values, window
            )
            
            self._cache_data(cache_key, volatility)
            return volatility
        
        except Exception as e:
            raise ValueError(f"Error calculating volatility for {symbol}: {str(e)}")
    
    def get_market_data_summary(self, symbol: str) -> Dict:
        """
        Get comprehensive market data summary
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Dictionary with market data summary
        """
        try:
            # Get current price
            current_price = self.get_current_price(symbol)
            
            # Get historical volatility
            hist_vol = self.calculate_historical_volatility(symbol)
            
            # Get risk-free rate
            rf_rate = self.get_risk_free_rate()
            
            # Get stock info
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                "symbol": symbol,
                "current_price": current_price,
                "historical_volatility": hist_vol,
                "risk_free_rate": rf_rate,
                "market_cap": info.get('marketCap', 0),
                "volume": info.get('volume', 0),
                "pe_ratio": info.get('trailingPE', 0),
                "beta": info.get('beta', 1.0),
                "sector": info.get('sector', 'Unknown'),
                "industry": info.get('industry', 'Unknown')
            }
        
        except Exception as e:
            raise ValueError(f"Error fetching market summary for {symbol}: {str(e)}")
    
    def get_available_expirations(self, symbol: str) -> List[str]:
        """
        Get available option expiration dates
        
        Args:
            symbol: Stock symbol
        
        Returns:
            List of expiration dates
        """
        try:
            ticker = yf.Ticker(symbol)
            return ticker.options
        except Exception as e:
            raise ValueError(f"Error fetching expirations for {symbol}: {str(e)}")
    
    def get_option_data(self, symbol: str, 
                       expiration_date: str,
                       option_type: str = "call") -> pd.DataFrame:
        """
        Get specific option data
        
        Args:
            symbol: Stock symbol
            expiration_date: Expiration date
            option_type: "call" or "put"
        
        Returns:
            DataFrame with option data
        """
        try:
            options_chain = self.get_options_chain(symbol, expiration_date)
            
            if option_type.lower() == "call":
                return options_chain["calls"]
            else:
                return options_chain["puts"]
        
        except Exception as e:
            raise ValueError(f"Error fetching {option_type} data for {symbol}: {str(e)}")
    
    def find_atm_options(self, symbol: str, 
                        expiration_date: str,
                        tolerance: float = 0.05) -> Dict:
        """
        Find at-the-money options
        
        Args:
            symbol: Stock symbol
            expiration_date: Expiration date
            tolerance: Moneyness tolerance
        
        Returns:
            Dictionary with ATM call and put
        """
        try:
            options_chain = self.get_options_chain(symbol, expiration_date)
            underlying_price = options_chain["underlying_price"]
            
            # Find ATM calls
            calls = options_chain["calls"]
            call_moneyness = abs(calls['strike'] - underlying_price) / underlying_price
            atm_call_idx = call_moneyness.idxmin()
            atm_call = calls.loc[atm_call_idx]
            
            # Find ATM puts
            puts = options_chain["puts"]
            put_moneyness = abs(puts['strike'] - underlying_price) / underlying_price
            atm_put_idx = put_moneyness.idxmin()
            atm_put = puts.loc[atm_put_idx]
            
            return {
                "call": atm_call.to_dict(),
                "put": atm_put.to_dict(),
                "underlying_price": underlying_price
            }
        
        except Exception as e:
            raise ValueError(f"Error finding ATM options for {symbol}: {str(e)}")
    
    def clear_cache(self):
        """Clear all cached data"""
        self.cache.clear()
    
    def get_cache_info(self) -> Dict:
        """Get information about cached data"""
        return {
            "cache_size": len(self.cache),
            "cache_keys": list(self.cache.keys()),
            "cache_timeout": self.cache_timeout
        } 