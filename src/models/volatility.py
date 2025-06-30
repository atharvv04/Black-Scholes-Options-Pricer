"""
Volatility calculation and analysis.

This module contains implementations for:
- Historical volatility calculation
- Implied volatility calculation
- Volatility surface analysis
- Volatility skew and term structure
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, List, Tuple, Dict
from sklearn.linear_model import LinearRegression
from .black_scholes import BlackScholesModel, OptionParameters
from .greeks import GreeksCalculator


class VolatilityCalculator:
    """
    Calculate historical and implied volatility
    """
    
    @staticmethod
    def historical_volatility(prices: Union[list, np.ndarray], 
                            window: int = 252) -> float:
        """
        Calculate historical volatility from price series
        
        Args:
            prices: Array of historical prices
            window: Number of trading days (default: 252 for annual)
        
        Returns:
            Annualized volatility
        """
        prices = np.array(prices)
        returns = np.log(prices[1:] / prices[:-1])
        
        # Calculate standard deviation and annualize
        volatility = np.std(returns) * np.sqrt(window)
        return volatility
    
    @staticmethod
    def rolling_volatility(prices: Union[list, np.ndarray], 
                          window: int = 30) -> np.ndarray:
        """
        Calculate rolling historical volatility
        
        Args:
            prices: Array of historical prices
            window: Rolling window size
        
        Returns:
            Array of rolling volatilities
        """
        prices = np.array(prices)
        returns = np.log(prices[1:] / prices[:-1])
        
        volatilities = []
        for i in range(window, len(returns)):
            window_returns = returns[i-window:i]
            vol = np.std(window_returns) * np.sqrt(252)  # Annualize
            volatilities.append(vol)
        
        return np.array(volatilities)
    
    @staticmethod
    def implied_volatility(option_price: float, 
                          S: float, K: float, T: float, r: float,
                          option_type: str = "call",
                          tolerance: float = 1e-5,
                          max_iterations: int = 100) -> float:
        """
        Calculate implied volatility using Newton-Raphson method
        
        Args:
            option_price: Market price of the option
            S: Current stock price
            K: Strike price
            T: Time to expiration
            r: Risk-free rate
            option_type: "call" or "put"
            tolerance: Convergence tolerance
            max_iterations: Maximum iterations
        
        Returns:
            Implied volatility
        """
        # Initial guess for volatility
        sigma = 0.3
        
        for i in range(max_iterations):
            # Create option parameters
            params = OptionParameters(S=S, K=K, T=T, r=r, sigma=sigma, 
                                   option_type=option_type)
            
            # Calculate option price and vega
            model = BlackScholesModel(params)
            calculated_price = model.price()
            vega = GreeksCalculator(model).vega()
            
            # Newton-Raphson update
            price_diff = calculated_price - option_price
            if abs(price_diff) < tolerance:
                return sigma
            
            sigma = sigma - price_diff / vega
            
            # Ensure volatility stays positive
            sigma = max(sigma, 1e-6)
        
        raise ValueError("Implied volatility calculation did not converge")
    
    @staticmethod
    def volatility_smile(strikes: List[float], 
                        implied_vols: List[float]) -> Dict[str, float]:
        """
        Analyze volatility smile characteristics
        
        Args:
            strikes: List of strike prices
            implied_vols: List of implied volatilities
        
        Returns:
            Dictionary with smile characteristics
        """
        if len(strikes) != len(implied_vols):
            raise ValueError("Strikes and implied volatilities must have same length")
        
        # Find ATM volatility (closest to 1.0 moneyness)
        moneyness = [s / strikes[0] for s in strikes]  # Assuming S = strikes[0]
        atm_index = np.argmin([abs(m - 1.0) for m in moneyness])
        atm_vol = implied_vols[atm_index]
        
        # Calculate skew (difference between high and low strike vols)
        skew = implied_vols[-1] - implied_vols[0]
        
        # Calculate kurtosis (curvature)
        if len(implied_vols) >= 3:
            # Use quadratic fit to measure curvature
            coeffs = np.polyfit(strikes, implied_vols, 2)
            curvature = coeffs[0] * 2  # Second derivative
        else:
            curvature = 0.0
        
        return {
            "atm_volatility": atm_vol,
            "skew": skew,
            "curvature": curvature,
            "min_volatility": min(implied_vols),
            "max_volatility": max(implied_vols),
            "volatility_range": max(implied_vols) - min(implied_vols)
        }
    
    @staticmethod
    def volatility_term_structure(expirations: List[float], 
                                implied_vols: List[float]) -> Dict[str, float]:
        """
        Analyze volatility term structure
        
        Args:
            expirations: List of expiration times
            implied_vols: List of implied volatilities
        
        Returns:
            Dictionary with term structure characteristics
        """
        if len(expirations) != len(implied_vols):
            raise ValueError("Expirations and implied volatilities must have same length")
        
        # Calculate forward volatility
        forward_vols = []
        for i in range(1, len(expirations)):
            T1, T2 = expirations[i-1], expirations[i]
            vol1, vol2 = implied_vols[i-1], implied_vols[i]
            
            # Forward volatility formula
            forward_vol = np.sqrt((vol2**2 * T2 - vol1**2 * T1) / (T2 - T1))
            forward_vols.append(forward_vol)
        
        # Calculate term structure slope
        if len(implied_vols) >= 2:
            slope = (implied_vols[-1] - implied_vols[0]) / (expirations[-1] - expirations[0])
        else:
            slope = 0.0
        
        return {
            "short_term_vol": implied_vols[0] if implied_vols else 0.0,
            "long_term_vol": implied_vols[-1] if implied_vols else 0.0,
            "term_slope": slope,
            "forward_vols": forward_vols,
            "volatility_convexity": np.std(implied_vols) if len(implied_vols) > 1 else 0.0
        }
    
    @staticmethod
    def volatility_forecast(historical_vols: List[float], 
                           method: str = "ewma") -> float:
        """
        Forecast volatility using various methods
        
        Args:
            historical_vols: List of historical volatilities
            method: Forecasting method ("ewma", "garch", "simple")
        
        Returns:
            Forecasted volatility
        """
        if method == "simple":
            return np.mean(historical_vols)
        
        elif method == "ewma":
            # Exponentially Weighted Moving Average
            lambda_param = 0.94  # RiskMetrics default
            weights = [(1 - lambda_param) * lambda_param**i 
                      for i in range(len(historical_vols))]
            weights = np.array(weights) / sum(weights)
            return np.sum(weights * historical_vols)
        
        elif method == "garch":
            # Simple GARCH(1,1) implementation
            omega = 0.000001
            alpha = 0.1
            beta = 0.8
            
            # Initialize
            forecast = historical_vols[0]**2
            
            for vol in historical_vols:
                forecast = omega + alpha * vol**2 + beta * forecast
            
            return np.sqrt(forecast)
        
        else:
            raise ValueError(f"Unknown forecasting method: {method}")


class VolatilitySurface:
    """Calculate and visualize implied volatility surface"""
    
    def __init__(self):
        self.surface_data = []
    
    def add_market_data(self, strikes: List[float], expirations: List[float], 
                       implied_vols: List[List[float]]):
        """
        Add market implied volatility data
        
        Args:
            strikes: List of strike prices
            expirations: List of expiration times
            implied_vols: 2D array of implied volatilities
        """
        for i, strike in enumerate(strikes):
            for j, expiration in enumerate(expirations):
                self.surface_data.append({
                    'strike': strike,
                    'expiration': expiration,
                    'implied_vol': implied_vols[i][j]
                })
    
    def interpolate_surface(self, strike_range: Tuple[float, float], 
                          expiration_range: Tuple[float, float],
                          n_points: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Interpolate volatility surface
        
        Returns:
            Tuple of (strikes_grid, expirations_grid, implied_vols_grid)
        """
        if not self.surface_data:
            raise ValueError("No surface data available")
        
        # Extract data
        strikes = [d['strike'] for d in self.surface_data]
        expirations = [d['expiration'] for d in self.surface_data]
        implied_vols = [d['implied_vol'] for d in self.surface_data]
        
        # Create grid
        strike_grid = np.linspace(strike_range[0], strike_range[1], n_points)
        expiration_grid = np.linspace(expiration_range[0], expiration_range[1], n_points)
        strikes_mesh, expirations_mesh = np.meshgrid(strike_grid, expiration_grid)
        
        # Interpolate using scipy
        try:
            from scipy.interpolate import griddata
            points = np.column_stack((strikes, expirations))
            implied_vols_grid = griddata(points, implied_vols, 
                                       (strikes_mesh, expirations_mesh), 
                                       method='cubic')
        except ImportError:
            # Fallback to simple interpolation
            implied_vols_grid = np.zeros_like(strikes_mesh)
            for i in range(n_points):
                for j in range(n_points):
                    # Find nearest neighbors
                    distances = [(abs(s - strikes_mesh[i, j]) + 
                                abs(e - expirations_mesh[i, j])) 
                               for s, e in zip(strikes, expirations)]
                    min_idx = np.argmin(distances)
                    implied_vols_grid[i, j] = implied_vols[min_idx]
        
        return strikes_mesh, expirations_mesh, implied_vols_grid
    
    def calculate_skew(self, expiration: float) -> float:
        """Calculate volatility skew for a given expiration"""
        # Filter data for given expiration
        exp_data = [d for d in self.surface_data if abs(d['expiration'] - expiration) < 0.01]
        
        if len(exp_data) < 3:
            return 0.0
        
        # Sort by strike price
        exp_data.sort(key=lambda x: x['strike'])
        
        # Calculate skew (difference between high and low strike vols)
        low_strike_vol = exp_data[0]['implied_vol']
        high_strike_vol = exp_data[-1]['implied_vol']
        
        return high_strike_vol - low_strike_vol
    
    def calculate_term_structure(self, strike: float) -> List[Tuple[float, float]]:
        """Calculate volatility term structure for a given strike"""
        # Filter data for given strike
        strike_data = [d for d in self.surface_data if abs(d['strike'] - strike) < 0.01]
        
        if not strike_data:
            return []
        
        # Sort by expiration
        strike_data.sort(key=lambda x: x['expiration'])
        
        return [(d['expiration'], d['implied_vol']) for d in strike_data] 