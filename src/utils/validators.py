"""
Input validation utilities for the Black-Scholes Options Pricer.

This module provides comprehensive validation for all user inputs to ensure
data integrity and prevent calculation errors.
"""

from typing import Union, Tuple, Dict, Any
import re


class InputValidator:
    """Validate user inputs for option pricing"""
    
    @staticmethod
    def validate_stock_price(price: Union[int, float]) -> Tuple[bool, str]:
        """Validate stock price input"""
        if not isinstance(price, (int, float)):
            return False, "Stock price must be a number"
        if price <= 0:
            return False, "Stock price must be positive"
        if price > 10000:
            return False, "Stock price seems unusually high (>$10,000)"
        return True, ""
    
    @staticmethod
    def validate_strike_price(strike: Union[int, float]) -> Tuple[bool, str]:
        """Validate strike price input"""
        if not isinstance(strike, (int, float)):
            return False, "Strike price must be a number"
        if strike <= 0:
            return False, "Strike price must be positive"
        if strike > 10000:
            return False, "Strike price seems unusually high (>$10,000)"
        return True, ""
    
    @staticmethod
    def validate_time_to_expiry(time: Union[int, float]) -> Tuple[bool, str]:
        """Validate time to expiration input"""
        if not isinstance(time, (int, float)):
            return False, "Time to expiration must be a number"
        if time <= 0:
            return False, "Time to expiration must be positive"
        if time > 10:
            return False, "Time to expiration seems unusually long (>10 years)"
        return True, ""
    
    @staticmethod
    def validate_volatility(vol: Union[int, float]) -> Tuple[bool, str]:
        """Validate volatility input"""
        if not isinstance(vol, (int, float)):
            return False, "Volatility must be a number"
        if vol <= 0:
            return False, "Volatility must be positive"
        if vol > 2:
            return False, "Volatility seems unusually high (>200%)"
        return True, ""
    
    @staticmethod
    def validate_interest_rate(rate: Union[int, float]) -> Tuple[bool, str]:
        """Validate interest rate input"""
        if not isinstance(rate, (int, float)):
            return False, "Interest rate must be a number"
        if rate < -0.1:
            return False, "Interest rate seems unusually low (<-10%)"
        if rate > 0.5:
            return False, "Interest rate seems unusually high (>50%)"
        return True, ""
    
    @staticmethod
    def validate_option_type(option_type: str) -> Tuple[bool, str]:
        """Validate option type input"""
        if not isinstance(option_type, str):
            return False, "Option type must be a string"
        if option_type.lower() not in ["call", "put"]:
            return False, "Option type must be 'call' or 'put'"
        return True, ""
    
    @staticmethod
    def validate_stock_symbol(symbol: str) -> Tuple[bool, str]:
        """Validate stock symbol input"""
        if not isinstance(symbol, str):
            return False, "Stock symbol must be a string"
        if not symbol.strip():
            return False, "Stock symbol cannot be empty"
        if len(symbol) > 10:
            return False, "Stock symbol seems too long"
        # Check for valid characters (letters, numbers, dots)
        if not re.match(r'^[A-Za-z0-9.]+$', symbol):
            return False, "Stock symbol contains invalid characters"
        return True, ""
    
    @staticmethod
    def validate_percentage(value: Union[int, float], 
                          min_value: float = 0.0, 
                          max_value: float = 100.0) -> Tuple[bool, str]:
        """Validate percentage input"""
        if not isinstance(value, (int, float)):
            return False, "Percentage must be a number"
        if value < min_value:
            return False, f"Percentage must be at least {min_value}%"
        if value > max_value:
            return False, f"Percentage must be at most {max_value}%"
        return True, ""
    
    @staticmethod
    def validate_quantity(quantity: Union[int, float]) -> Tuple[bool, str]:
        """Validate quantity input"""
        if not isinstance(quantity, (int, float)):
            return False, "Quantity must be a number"
        if quantity <= 0:
            return False, "Quantity must be positive"
        if quantity > 1000000:
            return False, "Quantity seems unusually high"
        return True, ""
    
    @staticmethod
    def validate_date_format(date_str: str) -> Tuple[bool, str]:
        """Validate date format (YYYY-MM-DD)"""
        if not isinstance(date_str, str):
            return False, "Date must be a string"
        
        # Check format YYYY-MM-DD
        if not re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
            return False, "Date must be in YYYY-MM-DD format"
        
        try:
            from datetime import datetime
            datetime.strptime(date_str, '%Y-%m-%d')
            return True, ""
        except ValueError:
            return False, "Invalid date"
    
    @staticmethod
    def validate_option_parameters(params: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate all option parameters at once"""
        validations = [
            ("S", InputValidator.validate_stock_price),
            ("K", InputValidator.validate_strike_price),
            ("T", InputValidator.validate_time_to_expiry),
            ("r", InputValidator.validate_interest_rate),
            ("sigma", InputValidator.validate_volatility),
            ("option_type", InputValidator.validate_option_type)
        ]
        
        for param_name, validator in validations:
            if param_name in params:
                is_valid, error_msg = validator(params[param_name])
                if not is_valid:
                    return False, f"{param_name}: {error_msg}"
        
        return True, ""
    
    @staticmethod
    def validate_market_data_params(params: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate market data parameters"""
        if "symbol" in params:
            is_valid, error_msg = InputValidator.validate_stock_symbol(params["symbol"])
            if not is_valid:
                return False, error_msg
        
        if "period" in params:
            valid_periods = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
            if params["period"] not in valid_periods:
                return False, f"Period must be one of: {', '.join(valid_periods)}"
        
        return True, ""
    
    @staticmethod
    def sanitize_input(value: Any, input_type: str = "float") -> Any:
        """Sanitize input values"""
        if input_type == "float":
            try:
                return float(value)
            except (ValueError, TypeError):
                return None
        elif input_type == "int":
            try:
                return int(value)
            except (ValueError, TypeError):
                return None
        elif input_type == "str":
            return str(value).strip()
        else:
            return value
    
    @staticmethod
    def format_error_message(errors: Dict[str, str]) -> str:
        """Format validation errors into a readable message"""
        if not errors:
            return ""
        
        error_list = [f"{field}: {error}" for field, error in errors.items()]
        return "Validation errors:\n" + "\n".join(error_list)


class ParameterValidator:
    """Advanced parameter validation with business logic"""
    
    @staticmethod
    def validate_black_scholes_params(S: float, K: float, T: float, 
                                    r: float, sigma: float, 
                                    option_type: str) -> Tuple[bool, str]:
        """Validate Black-Scholes parameters with business logic"""
        # Basic validations
        basic_validations = [
            ("S", InputValidator.validate_stock_price(S)),
            ("K", InputValidator.validate_strike_price(K)),
            ("T", InputValidator.validate_time_to_expiry(T)),
            ("r", InputValidator.validate_interest_rate(r)),
            ("sigma", InputValidator.validate_volatility(sigma)),
            ("option_type", InputValidator.validate_option_type(option_type))
        ]
        
        for param_name, (is_valid, error_msg) in basic_validations:
            if not is_valid:
                return False, f"{param_name}: {error_msg}"
        
        # Business logic validations
        if S <= 0 or K <= 0:
            return False, "Stock price and strike price must be positive"
        
        if T <= 0:
            return False, "Time to expiration must be positive"
        
        if sigma <= 0:
            return False, "Volatility must be positive"
        
        # Check for extreme values that might cause numerical issues
        if sigma > 1.0:  # > 100%
            return False, "Volatility seems extremely high (>100%)"
        
        if T > 5.0:  # > 5 years
            return False, "Time to expiration seems very long (>5 years)"
        
        return True, ""
    
    @staticmethod
    def validate_greeks_calculation_params(params: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate parameters for Greeks calculations"""
        required_params = ["S", "K", "T", "r", "sigma", "option_type"]
        
        for param in required_params:
            if param not in params:
                return False, f"Missing required parameter: {param}"
        
        return ParameterValidator.validate_black_scholes_params(
            params["S"], params["K"], params["T"], 
            params["r"], params["sigma"], params["option_type"]
        )
    
    @staticmethod
    def validate_volatility_calculation_params(prices: list, 
                                             window: int) -> Tuple[bool, str]:
        """Validate parameters for volatility calculations"""
        if not isinstance(prices, (list, tuple)):
            return False, "Prices must be a list or array"
        
        if len(prices) < 2:
            return False, "Need at least 2 price points for volatility calculation"
        
        if not all(isinstance(p, (int, float)) for p in prices):
            return False, "All prices must be numbers"
        
        if not all(p > 0 for p in prices):
            return False, "All prices must be positive"
        
        if window <= 0:
            return False, "Window size must be positive"
        
        if window > len(prices):
            return False, "Window size cannot exceed number of price points"
        
        return True, ""
    
    @staticmethod
    def validate_moneyness_ratio(S: float, K: float) -> Tuple[bool, str]:
        """Validate moneyness ratio (S/K)"""
        if K <= 0:
            return False, "Strike price must be positive"
        
        moneyness = S / K
        
        if moneyness < 0.1:  # S < 0.1K
            return False, "Stock price seems extremely low relative to strike"
        
        if moneyness > 10:  # S > 10K
            return False, "Stock price seems extremely high relative to strike"
        
        return True, ""
    
    @staticmethod
    def validate_implied_volatility_params(option_price: float, 
                                         S: float, K: float, T: float, 
                                         r: float, option_type: str) -> Tuple[bool, str]:
        """Validate parameters for implied volatility calculation"""
        # Basic validations
        if option_price <= 0:
            return False, "Option price must be positive"
        
        # Validate other parameters
        is_valid, error_msg = ParameterValidator.validate_black_scholes_params(
            S, K, T, r, 0.3, option_type  # Use dummy volatility for validation
        )
        
        if not is_valid:
            return False, error_msg
        
        # Check if option price is reasonable
        intrinsic_value = max(0, S - K) if option_type == "call" else max(0, K - S)
        if option_price < intrinsic_value:
            return False, "Option price cannot be less than intrinsic value"
        
        return True, "" 