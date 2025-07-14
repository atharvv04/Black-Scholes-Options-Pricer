#!/usr/bin/env python3
"""
Test script to verify all imports work correctly.
"""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test all imports"""
    try:
        print("Testing imports...")
        
        # Test core models
        from models.black_scholes import BlackScholesModel, OptionParameters
        print("✅ Black-Scholes model imported successfully")
        
        from models.greeks import GreeksCalculator
        print("✅ Greeks calculator imported successfully")
        
        from models.volatility import VolatilityCalculator
        print("✅ Volatility calculator imported successfully")
        
        # Test utils
        from utils.data_fetcher import MarketDataFetcher
        print("✅ Market data fetcher imported successfully")
        
        from utils.validators import InputValidator
        print("✅ Input validator imported successfully")
        
        from utils.calculators import OptionCalculators
        print("✅ Option calculators imported successfully")
        
        # Test visualization
        from visualization.charts import OptionCharts
        print("✅ Option charts imported successfully")
        
        # Test web app
        from web.app import BlackScholesApp
        print("✅ Web app imported successfully")
        
        print("\n🎉 All imports successful!")
        return True
        
    except Exception as e:
        print(f"❌ Import error: {str(e)}")
        return False

def test_basic_functionality():
    """Test basic functionality"""
    try:
        print("\nTesting basic functionality...")
        
        # Import the classes we need
        from models.black_scholes import BlackScholesModel, OptionParameters
        from models.greeks import GreeksCalculator
        from models.volatility import VolatilityCalculator
        
        # Create option parameters
        params = OptionParameters(
            S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2, option_type="call"
        )
        
        # Create model and price option
        model = BlackScholesModel(params)
        price = model.price()
        print(f"✅ Option pricing works: ${price:.4f}")
        
        # Calculate Greeks
        greeks_calc = GreeksCalculator(model)
        greeks = greeks_calc.all_greeks()
        print(f"✅ Greeks calculation works: Delta = {greeks['delta']:.4f}")
        
        # Test volatility calculation
        hist_vol = VolatilityCalculator.historical_volatility([100, 101, 99, 102, 98])
        print(f"✅ Volatility calculation works: {hist_vol:.1%}")
        
        print("🎉 Basic functionality test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Functionality error: {str(e)}")
        return False

if __name__ == "__main__":
    print("Black-Scholes Options Pricer - Import Test")
    print("=" * 50)
    
    imports_ok = test_imports()
    functionality_ok = test_basic_functionality()
    
    if imports_ok and functionality_ok:
        print("\n✅ All tests passed! The application is ready to run.")
        print("Run 'python run_app.py' to start the web application.")
    else:
        print("\n❌ Some tests failed. Please check the errors above.") 