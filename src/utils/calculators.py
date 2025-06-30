"""
Option calculation utilities.

This module provides utility functions for option calculations including
breakeven points, profit/loss calculations, and option strategies.
"""

import numpy as np
from typing import List, Dict, Union


class OptionCalculators:
    """Utility calculators for option analysis"""
    
    @staticmethod
    def breakeven_points(S: float, K: float, premium: float, 
                        option_type: str) -> List[float]:
        """
        Calculate breakeven points for an option
        
        Args:
            S: Current stock price
            K: Strike price
            premium: Option premium
            option_type: "call" or "put"
        
        Returns:
            List of breakeven stock prices
        """
        if option_type == "call":
            # For calls: breakeven = strike + premium
            return [K + premium]
        else:
            # For puts: breakeven = strike - premium
            return [K - premium]
    
    @staticmethod
    def profit_loss_at_expiry(stock_prices: List[float], K: float, 
                             premium: float, option_type: str) -> List[float]:
        """
        Calculate profit/loss at expiration for different stock prices
        
        Args:
            stock_prices: List of possible stock prices at expiration
            K: Strike price
            premium: Option premium paid
            option_type: "call" or "put"
        
        Returns:
            List of profit/loss values
        """
        payoffs = []
        
        for S in stock_prices:
            if option_type == "call":
                payoff = max(0, S - K) - premium
            else:  # put option
                payoff = max(0, K - S) - premium
            payoffs.append(payoff)
        
        return payoffs
    
    @staticmethod
    def option_strategies() -> Dict[str, Dict]:
        """
        Define common option strategies
        
        Returns:
            Dictionary of strategy definitions
        """
        return {
            "long_call": {
                "name": "Long Call",
                "description": "Buy a call option",
                "max_loss": "Premium paid",
                "max_profit": "Unlimited",
                "breakeven": "Strike + Premium"
            },
            "long_put": {
                "name": "Long Put",
                "description": "Buy a put option",
                "max_loss": "Premium paid",
                "max_profit": "Strike - Premium",
                "breakeven": "Strike - Premium"
            },
            "covered_call": {
                "name": "Covered Call",
                "description": "Sell call + own stock",
                "max_loss": "Stock price - Strike + Premium",
                "max_profit": "Premium + (Strike - Stock price)",
                "breakeven": "Stock price - Premium"
            },
            "protective_put": {
                "name": "Protective Put",
                "description": "Buy put + own stock",
                "max_loss": "Premium paid",
                "max_profit": "Unlimited",
                "breakeven": "Stock price + Premium"
            },
            "straddle": {
                "name": "Straddle",
                "description": "Buy call + buy put (same strike)",
                "max_loss": "Total premium paid",
                "max_profit": "Unlimited",
                "breakeven": "Strike ± Total premium"
            }
        }
    
    @staticmethod
    def calculate_portfolio_value(positions: Dict[str, Dict], 
                                current_prices: Dict[str, float]) -> float:
        """
        Calculate total portfolio value
        
        Args:
            positions: Dictionary of positions {symbol: {quantity, type}}
            current_prices: Dictionary of current prices {symbol: price}
        
        Returns:
            Total portfolio value
        """
        total_value = 0.0
        
        for symbol, position in positions.items():
            if symbol in current_prices:
                quantity = position.get('quantity', 0)
                price = current_prices[symbol]
                total_value += quantity * price
        
        return total_value
    
    @staticmethod
    def calculate_portfolio_pnl(positions: Dict[str, Dict], 
                              current_prices: Dict[str, float],
                              entry_prices: Dict[str, float]) -> float:
        """
        Calculate portfolio profit/loss
        
        Args:
            positions: Dictionary of positions
            current_prices: Current market prices
            entry_prices: Entry prices for positions
        
        Returns:
            Total profit/loss
        """
        total_pnl = 0.0
        
        for symbol, position in positions.items():
            if symbol in current_prices and symbol in entry_prices:
                quantity = position.get('quantity', 0)
                current_price = current_prices[symbol]
                entry_price = entry_prices[symbol]
                pnl = quantity * (current_price - entry_price)
                total_pnl += pnl
        
        return total_pnl
    
    @staticmethod
    def calculate_risk_metrics(portfolio_value: float, 
                             portfolio_pnl: float,
                             max_loss: float) -> Dict[str, float]:
        """
        Calculate portfolio risk metrics
        
        Args:
            portfolio_value: Current portfolio value
            portfolio_pnl: Current profit/loss
            max_loss: Maximum possible loss
        
        Returns:
            Dictionary of risk metrics
        """
        return {
            "total_return": portfolio_pnl / portfolio_value if portfolio_value > 0 else 0,
            "max_loss_ratio": max_loss / portfolio_value if portfolio_value > 0 else 0,
            "risk_reward_ratio": abs(portfolio_pnl / max_loss) if max_loss > 0 else 0,
            "profit_factor": portfolio_pnl / max_loss if max_loss > 0 else 0
        } 