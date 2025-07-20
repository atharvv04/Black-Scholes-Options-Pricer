#!/usr/bin/env python3
"""
Launcher script for the Black-Scholes Options Pricer.

This script properly sets up the Python path and launches the Streamlit application.
"""

import sys
import os
import subprocess

def setup_environment():
    """Set up the Python environment for the application"""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Add the src directory to Python path
    src_dir = os.path.join(script_dir, 'src')
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    
    # Change to the project directory
    os.chdir(script_dir)
    
    return script_dir

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'streamlit', 'numpy', 'scipy', 'pandas', 
        'matplotlib', 'plotly', 'yfinance', 'scikit-learn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} is missing")
    
    if missing_packages:
        print(f"\nInstalling missing packages: {', '.join(missing_packages)}")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                *missing_packages
            ])
            print("✅ All dependencies installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error installing dependencies: {e}")
            return False
    
    return True

def run_tests():
    """Run basic tests to ensure everything works"""
    print("\n🧪 Running basic tests...")
    
    try:
        # Test imports
        from models.black_scholes import BlackScholesModel, OptionParameters
        from models.greeks import GreeksCalculator
        from utils.data_fetcher import MarketDataFetcher
        from visualization.charts import OptionCharts
        
        # Test basic functionality
        params = OptionParameters(
            S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2, option_type="call"
        )
        model = BlackScholesModel(params)
        price = model.price()
        
        print(f"✅ Basic test passed - Option price: ${price:.4f}")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False

def launch_app():
    """Launch the Streamlit application"""
    print("\n🚀 Launching Black-Scholes Options Pricer...")
    print("📊 The application will open in your web browser")
    print("🔗 If it doesn't open automatically, go to: http://localhost:8501")
    print()
    
    try:
        # Run the Streamlit app using the simple version
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "simple_app.py",
            "--server.port=8501",
            "--server.address=localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error launching application: {str(e)}")

def main():
    """Main function"""
    print("Black-Scholes Options Pricer - Launcher")
    print("=" * 50)
    
    # Set up environment
    setup_environment()
    
    # Check dependencies
    if not check_dependencies():
        print("❌ Failed to install dependencies. Please install them manually:")
        print("pip install -r requirements.txt")
        return
    
    # Run tests
    if not run_tests():
        print("❌ Tests failed. Please check the errors above.")
        return
    
    # Launch app
    launch_app()

if __name__ == "__main__":
    main() 