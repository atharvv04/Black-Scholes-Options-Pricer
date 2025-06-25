"""
Black-Scholes option pricing model implementation.

This module contains the core Black-Scholes formula implementation for pricing
European-style options with comprehensive parameter validation and utility methods.
"""

import numpy as np
import scipy.stats as si
from typing import Union, Tuple, Optional
from dataclasses import dataclass


@dataclass
class OptionParameters:
    """Container for option pricing parameters"""
    S: float  # Current stock price
    K: float  # Strike price
    T: float  # Time to expiration (years)
    r: float  # Risk-free interest rate
    sigma: float  # Volatility
    option_type: str = "call"  # "call" or "put"
    
    def __post_init__(self):
        """Validate parameters after initialization"""
        if self.S <= 0:
            raise ValueError("Stock price must be positive")
        if self.K <= 0:
            raise ValueError("Strike price must be positive")
        if self.T <= 0:
            raise ValueError("Time to expiration must be positive")
        if self.sigma <= 0:
            raise ValueError("Volatility must be positive")
        if self.option_type not in ["call", "put"]:
            raise ValueError("Option type must be 'call' or 'put'")


class BlackScholesModel:
    """
    Implementation of the Black-Scholes option pricing model
    
    The Black-Scholes formula for European options:
    
    Call Option: C = S*N(d1) - K*e^(-rT)*N(d2)
    Put Option:  P = K*e^(-rT)*N(-d2) - S*N(-d1)
    
    Where:
    d1 = (ln(S/K) + (r + σ²/2)T) / (σ√T)
    d2 = d1 - σ√T
    S = Current stock price
    K = Strike price
    T = Time to expiration
    r = Risk-free rate
    σ = Volatility
    N() = Cumulative normal distribution
    """
    
    def __init__(self, params: OptionParameters):
        self.params = params
    
    def _calculate_d1_d2(self) -> Tuple[float, float]:
        """Calculate d1 and d2 parameters"""
        S, K, T, r, sigma = (
            self.params.S, self.params.K, self.params.T, 
            self.params.r, self.params.sigma
        )
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        return d1, d2
    
    def price(self) -> float:
        """Calculate option price using Black-Scholes formula"""
        d1, d2 = self._calculate_d1_d2()
        S, K, T, r = self.params.S, self.params.K, self.params.T, self.params.r
        
        if self.params.option_type == "call":
            price = S * si.norm.cdf(d1) - K * np.exp(-r * T) * si.norm.cdf(d2)
        else:  # put option
            price = K * np.exp(-r * T) * si.norm.cdf(-d2) - S * si.norm.cdf(-d1)
        
        return price
    
    def is_in_the_money(self) -> bool:
        """Check if option is in the money"""
        if self.params.option_type == "call":
            return self.params.S > self.params.K
        else:
            return self.params.S < self.params.K
    
    def is_at_the_money(self) -> bool:
        """Check if option is at the money"""
        return abs(self.params.S - self.params.K) < 0.01
    
    def is_out_of_the_money(self) -> bool:
        """Check if option is out of the money"""
        return not (self.is_in_the_money() or self.is_at_the_money())
    
    def intrinsic_value(self) -> float:
        """Calculate intrinsic value of the option"""
        if self.params.option_type == "call":
            return max(0, self.params.S - self.params.K)
        else:
            return max(0, self.params.K - self.params.S)
    
    def time_value(self) -> float:
        """Calculate time value of the option"""
        return self.price() - self.intrinsic_value()
    
    def moneyness(self) -> float:
        """Calculate moneyness (S/K ratio)"""
        return self.params.S / self.params.K
    
    def get_option_status(self) -> str:
        """Get descriptive status of the option"""
        if self.is_at_the_money():
            return "At-the-Money"
        elif self.is_in_the_money():
            return "In-the-Money"
        else:
            return "Out-of-the-Money"
    
    def calculate_probability_of_profit(self) -> float:
        """Calculate probability of profit at expiration"""
        d2 = self._calculate_d1_d2()[1]
        
        if self.params.option_type == "call":
            return si.norm.cdf(d2)
        else:
            return si.norm.cdf(-d2)
    
    def calculate_expected_value(self) -> float:
        """Calculate expected value at expiration"""
        probability = self.calculate_probability_of_profit()
        
        if self.params.option_type == "call":
            expected_payoff = max(0, self.params.S - self.params.K)
        else:
            expected_payoff = max(0, self.params.K - self.params.S)
        
        return probability * expected_payoff - self.price()
    
    def to_dict(self) -> dict:
        """Convert model to dictionary representation"""
        return {
            "stock_price": self.params.S,
            "strike_price": self.params.K,
            "time_to_expiration": self.params.T,
            "risk_free_rate": self.params.r,
            "volatility": self.params.sigma,
            "option_type": self.params.option_type,
            "price": self.price(),
            "intrinsic_value": self.intrinsic_value(),
            "time_value": self.time_value(),
            "moneyness": self.moneyness(),
            "status": self.get_option_status(),
            "probability_of_profit": self.calculate_probability_of_profit()
        } 