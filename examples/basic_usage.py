"""
Basic usage example for the Black-Scholes Options Pricer.

This script demonstrates how to use the core components of the application.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.black_scholes import BlackScholesModel, OptionParameters
from src.models.greeks import GreeksCalculator
from src.utils.data_fetcher import MarketDataFetcher
from src.visualization.charts import OptionCharts


def basic_option_pricing_example():
    """Demonstrate basic option pricing"""
    print("=== Basic Option Pricing Example ===")
    
    # Create option parameters
    params = OptionParameters(
        S=100.0,      # Current stock price
        K=100.0,      # Strike price
        T=1.0,        # Time to expiration (years)
        r=0.05,       # Risk-free rate
        sigma=0.2,    # Volatility
        option_type="call"
    )
    
    # Create model and price option
    model = BlackScholesModel(params)
    price = model.price()
    
    print(f"Call Option Price: ${price:.4f}")
    print(f"Intrinsic Value: ${model.intrinsic_value():.4f}")
    print(f"Time Value: ${model.time_value():.4f}")
    print(f"Option Status: {model.get_option_status()}")
    print(f"Moneyness: {model.moneyness():.2f}")
    print(f"Probability of Profit: {model.calculate_probability_of_profit():.1%}")
    print()


def greeks_analysis_example():
    """Demonstrate Greeks calculation"""
    print("=== Greeks Analysis Example ===")
    
    # Create option parameters
    params = OptionParameters(
        S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2, option_type="call"
    )
    model = BlackScholesModel(params)
    
    # Calculate Greeks
    greeks_calc = GreeksCalculator(model)
    greeks = greeks_calc.all_greeks()
    
    print("Option Greeks:")
    print(f"Delta: {greeks['delta']:.4f}")
    print(f"Gamma: {greeks['gamma']:.4f}")
    print(f"Theta: {greeks['theta']:.4f}")
    print(f"Vega: {greeks['vega']:.4f}")
    print(f"Rho: {greeks['rho']:.4f}")
    print()


def market_data_example():
    """Demonstrate market data integration"""
    print("=== Market Data Example ===")
    
    try:
        # Create data fetcher
        fetcher = MarketDataFetcher()
        
        # Fetch market data for AAPL
        symbol = "AAPL"
        current_price = fetcher.get_current_price(symbol)
        hist_vol = fetcher.calculate_historical_volatility(symbol)
        rf_rate = fetcher.get_risk_free_rate()
        
        print(f"Market Data for {symbol}:")
        print(f"Current Price: ${current_price:.2f}")
        print(f"Historical Volatility: {hist_vol:.1%}")
        print(f"Risk-free Rate: {rf_rate:.1%}")
        
        # Price option with market data
        params = OptionParameters(
            S=current_price,
            K=current_price,  # At-the-money
            T=0.25,           # 3 months
            r=rf_rate,
            sigma=hist_vol,
            option_type="call"
        )
        
        model = BlackScholesModel(params)
        price = model.price()
        
        print(f"ATM Call Option Price: ${price:.4f}")
        print()
        
    except Exception as e:
        print(f"Error fetching market data: {str(e)}")
        print()


def sensitivity_analysis_example():
    """Demonstrate sensitivity analysis"""
    print("=== Sensitivity Analysis Example ===")
    
    # Create base model
    params = OptionParameters(
        S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2, option_type="call"
    )
    model = BlackScholesModel(params)
    
    # Test different stock prices
    stock_prices = [80, 90, 100, 110, 120]
    prices = []
    
    for S in stock_prices:
        params.S = S
        new_model = BlackScholesModel(params)
        prices.append(new_model.price())
    
    print("Option Prices at Different Stock Prices:")
    for S, price in zip(stock_prices, prices):
        print(f"Stock Price: ${S}, Option Price: ${price:.4f}")
    print()


def comparison_example():
    """Compare call and put options"""
    print("=== Call vs Put Comparison ===")
    
    # Base parameters
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    
    # Call option
    call_params = OptionParameters(S=S, K=K, T=T, r=r, sigma=sigma, option_type="call")
    call_model = BlackScholesModel(call_params)
    call_price = call_model.price()
    
    # Put option
    put_params = OptionParameters(S=S, K=K, T=T, r=r, sigma=sigma, option_type="put")
    put_model = BlackScholesModel(put_params)
    put_price = put_model.price()
    
    print(f"Call Option Price: ${call_price:.4f}")
    print(f"Put Option Price: ${put_price:.4f}")
    print(f"Call Intrinsic Value: ${call_model.intrinsic_value():.4f}")
    print(f"Put Intrinsic Value: ${put_model.intrinsic_value():.4f}")
    print()


def main():
    """Run all examples"""
    print("Black-Scholes Options Pricer - Basic Usage Examples")
    print("=" * 50)
    print()
    
    # Run examples
    basic_option_pricing_example()
    greeks_analysis_example()
    market_data_example()
    sensitivity_analysis_example()
    comparison_example()
    
    print("Examples completed successfully!")


if __name__ == "__main__":
    main() 