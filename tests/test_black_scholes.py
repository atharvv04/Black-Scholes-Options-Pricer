"""
Basic tests for the Black-Scholes model implementation.
"""

import pytest
import numpy as np
from src.models.black_scholes import BlackScholesModel, OptionParameters


class TestBlackScholesModel:
    """Test cases for Black-Scholes model"""
    
    def setup_method(self):
        """Setup test parameters"""
        self.params = OptionParameters(
            S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2, option_type="call"
        )
        self.model = BlackScholesModel(self.params)
    
    def test_call_option_pricing(self):
        """Test call option pricing"""
        price = self.model.price()
        assert 0 < price < 20  # Reasonable range for at-the-money call
        assert isinstance(price, float)
    
    def test_put_option_pricing(self):
        """Test put option pricing"""
        put_params = OptionParameters(
            S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2, option_type="put"
        )
        put_model = BlackScholesModel(put_params)
        price = put_model.price()
        assert 0 < price < 20
        assert isinstance(price, float)
    
    def test_parameter_validation(self):
        """Test parameter validation"""
        # Test negative stock price
        with pytest.raises(ValueError):
            invalid_params = OptionParameters(
                S=-100.0, K=100.0, T=1.0, r=0.05, sigma=0.2, option_type="call"
            )
            BlackScholesModel(invalid_params)
        
        # Test invalid option type
        with pytest.raises(ValueError):
            invalid_params = OptionParameters(
                S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2, option_type="invalid"
            )
            BlackScholesModel(invalid_params)
    
    def test_intrinsic_value(self):
        """Test intrinsic value calculation"""
        # In-the-money call
        itm_params = OptionParameters(
            S=110.0, K=100.0, T=1.0, r=0.05, sigma=0.2, option_type="call"
        )
        itm_model = BlackScholesModel(itm_params)
        assert itm_model.intrinsic_value() == 10.0
        
        # Out-of-the-money call
        otm_params = OptionParameters(
            S=90.0, K=100.0, T=1.0, r=0.05, sigma=0.2, option_type="call"
        )
        otm_model = BlackScholesModel(otm_params)
        assert otm_model.intrinsic_value() == 0.0
    
    def test_moneyness(self):
        """Test moneyness calculation"""
        # In-the-money call
        itm_params = OptionParameters(
            S=110.0, K=100.0, T=1.0, r=0.05, sigma=0.2, option_type="call"
        )
        itm_model = BlackScholesModel(itm_params)
        assert itm_model.is_in_the_money() == True
        
        # Out-of-the-money call
        otm_params = OptionParameters(
            S=90.0, K=100.0, T=1.0, r=0.05, sigma=0.2, option_type="call"
        )
        otm_model = BlackScholesModel(otm_params)
        assert otm_model.is_in_the_money() == False
    
    def test_option_status(self):
        """Test option status determination"""
        # At-the-money
        atm_params = OptionParameters(
            S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2, option_type="call"
        )
        atm_model = BlackScholesModel(atm_params)
        assert atm_model.get_option_status() == "At-the-Money"
        
        # In-the-money
        itm_params = OptionParameters(
            S=110.0, K=100.0, T=1.0, r=0.05, sigma=0.2, option_type="call"
        )
        itm_model = BlackScholesModel(itm_params)
        assert itm_model.get_option_status() == "In-the-Money"
        
        # Out-of-the-money
        otm_params = OptionParameters(
            S=90.0, K=100.0, T=1.0, r=0.05, sigma=0.2, option_type="call"
        )
        otm_model = BlackScholesModel(otm_params)
        assert otm_model.get_option_status() == "Out-of-the-Money"
    
    def test_probability_of_profit(self):
        """Test probability of profit calculation"""
        prob = self.model.calculate_probability_of_profit()
        assert 0 <= prob <= 1
        assert isinstance(prob, float)
    
    def test_to_dict(self):
        """Test model to dictionary conversion"""
        result = self.model.to_dict()
        assert isinstance(result, dict)
        assert "price" in result
        assert "intrinsic_value" in result
        assert "time_value" in result
        assert "moneyness" in result
        assert "status" in result
        assert "probability_of_profit" in result


if __name__ == "__main__":
    pytest.main([__file__]) 