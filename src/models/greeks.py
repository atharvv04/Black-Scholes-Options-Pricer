"""
Greeks calculation for options pricing.

This module contains implementations for calculating option Greeks:
- Delta: Price sensitivity to underlying asset price
- Gamma: Delta sensitivity to underlying asset price  
- Theta: Price sensitivity to time decay
- Vega: Price sensitivity to volatility
- Rho: Price sensitivity to interest rate
"""

import numpy as np
import scipy.stats as si
from typing import Dict, Union
from .black_scholes import BlackScholesModel, OptionParameters


class GreeksCalculator:
    """
    Calculate option Greeks (Delta, Gamma, Theta, Vega, Rho)
    
    Greeks measure the sensitivity of option price to various factors:
    - Delta: Price sensitivity to underlying asset price
    - Gamma: Delta sensitivity to underlying asset price
    - Theta: Price sensitivity to time decay
    - Vega: Price sensitivity to volatility
    - Rho: Price sensitivity to interest rate
    """
    
    def __init__(self, model: BlackScholesModel):
        self.model = model
        self.params = model.params
    
    def _get_d1_d2(self) -> tuple:
        """Get d1 and d2 values"""
        return self.model._calculate_d1_d2()
    
    def delta(self) -> float:
        """Calculate option delta"""
        d1, _ = self._get_d1_d2()
        
        if self.params.option_type == "call":
            return si.norm.cdf(d1)
        else:  # put option
            return si.norm.cdf(d1) - 1
    
    def gamma(self) -> float:
        """Calculate option gamma (same for calls and puts)"""
        d1, _ = self._get_d1_d2()
        S, sigma, T = self.params.S, self.params.sigma, self.params.T
        
        return si.norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    def theta(self) -> float:
        """Calculate option theta (time decay)"""
        d1, d2 = self._get_d1_d2()
        S, K, T, r, sigma = (
            self.params.S, self.params.K, self.params.T, 
            self.params.r, self.params.sigma
        )
        
        theta_term1 = -S * si.norm.pdf(d1) * sigma / (2 * np.sqrt(T))
        
        if self.params.option_type == "call":
            theta_term2 = -r * K * np.exp(-r * T) * si.norm.cdf(d2)
        else:  # put option
            theta_term2 = r * K * np.exp(-r * T) * si.norm.cdf(-d2)
        
        return theta_term1 + theta_term2
    
    def vega(self) -> float:
        """Calculate option vega (same for calls and puts)"""
        d1, _ = self._get_d1_d2()
        S, T = self.params.S, self.params.T
        
        return S * si.norm.pdf(d1) * np.sqrt(T)
    
    def rho(self) -> float:
        """Calculate option rho"""
        _, d2 = self._get_d1_d2()
        K, T, r = self.params.K, self.params.T, self.params.r
        
        if self.params.option_type == "call":
            return K * T * np.exp(-r * T) * si.norm.cdf(d2)
        else:  # put option
            return -K * T * np.exp(-r * T) * si.norm.cdf(-d2)
    
    def all_greeks(self) -> Dict[str, float]:
        """Calculate all Greeks"""
        return {
            "delta": self.delta(),
            "gamma": self.gamma(),
            "theta": self.theta(),
            "vega": self.vega(),
            "rho": self.rho()
        }
    
    def greeks_sensitivity_analysis(self, parameter: str, 
                                  range_values: list) -> Dict[str, list]:
        """
        Calculate how Greeks change with parameter variations
        
        Args:
            parameter: Parameter to vary ("S", "K", "T", "r", "sigma")
            range_values: List of values to test
        
        Returns:
            Dictionary with Greeks for each parameter value
        """
        results = {
            "delta": [],
            "gamma": [],
            "theta": [],
            "vega": [],
            "rho": []
        }
        
        for value in range_values:
            # Create new parameters with varied value
            params_dict = self.params.__dict__.copy()
            params_dict[parameter] = value
            new_params = OptionParameters(**params_dict)
            new_model = BlackScholesModel(new_params)
            new_greeks = GreeksCalculator(new_model).all_greeks()
            
            for greek in results:
                results[greek].append(new_greeks[greek])
        
        return results
    
    def delta_hedge_ratio(self) -> float:
        """Calculate delta hedge ratio (number of shares to hedge)"""
        return -self.delta()
    
    def gamma_hedge_ratio(self) -> float:
        """Calculate gamma hedge ratio"""
        return -self.gamma()
    
    def vega_hedge_ratio(self) -> float:
        """Calculate vega hedge ratio"""
        return -self.vega()
    
    def portfolio_greeks(self, positions: Dict[str, int]) -> Dict[str, float]:
        """
        Calculate portfolio Greeks for multiple positions
        
        Args:
            positions: Dictionary of {option_id: quantity}
        
        Returns:
            Dictionary with portfolio Greeks
        """
        portfolio_greeks = {
            "delta": 0.0,
            "gamma": 0.0,
            "theta": 0.0,
            "vega": 0.0,
            "rho": 0.0
        }
        
        # This would need to be implemented with actual option positions
        # For now, return the current option's Greeks
        individual_greeks = self.all_greeks()
        
        for greek in portfolio_greeks:
            portfolio_greeks[greek] = individual_greeks[greek]
        
        return portfolio_greeks
    
    def risk_metrics(self) -> Dict[str, float]:
        """Calculate risk metrics based on Greeks"""
        greeks = self.all_greeks()
        
        return {
            "delta_exposure": abs(greeks["delta"]),
            "gamma_exposure": abs(greeks["gamma"]),
            "vega_exposure": abs(greeks["vega"]),
            "theta_decay": abs(greeks["theta"]),
            "rho_exposure": abs(greeks["rho"]),
            "total_risk": abs(greeks["delta"]) + abs(greeks["gamma"]) + 
                         abs(greeks["vega"]) + abs(greeks["theta"]) + abs(greeks["rho"])
        }
    
    def to_dict(self) -> dict:
        """Convert Greeks to dictionary representation"""
        greeks = self.all_greeks()
        risk_metrics = self.risk_metrics()
        
        return {
            **greeks,
            **risk_metrics,
            "delta_hedge_ratio": self.delta_hedge_ratio(),
            "gamma_hedge_ratio": self.gamma_hedge_ratio(),
            "vega_hedge_ratio": self.vega_hedge_ratio()
        } 