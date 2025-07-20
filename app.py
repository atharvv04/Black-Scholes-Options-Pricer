#!/usr/bin/env python3
"""
Main entry point for the Black-Scholes Options Pricer web application.

This file serves as the main entry point for running the Streamlit application
and handles all the necessary imports and setup.
"""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import the main app
from web.app import main

if __name__ == "__main__":
    main() 