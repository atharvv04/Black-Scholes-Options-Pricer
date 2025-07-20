#!/usr/bin/env python3
"""
Simple script to run the Black-Scholes Options Pricer web application.
"""

import subprocess
import sys
import os

def main():
    """Run the Streamlit application"""
    try:
        # Check if streamlit is installed
        import streamlit
        print("✅ Streamlit is installed")
    except ImportError:
        print("❌ Streamlit is not installed. Installing dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # Run the Streamlit app
    print("🚀 Starting Black-Scholes Options Pricer...")
    print("📊 The application will open in your web browser")
    print("🔗 If it doesn't open automatically, go to: http://localhost:8501")
    print()
    
    # Change to the project directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Run streamlit using the main app.py file
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", 
        "app.py", 
        "--server.port=8501",
        "--server.address=localhost"
    ])

if __name__ == "__main__":
    main() 