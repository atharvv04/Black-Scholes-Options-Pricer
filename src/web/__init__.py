"""
Web application components for the Black-Scholes Options Pricer.

This package contains:
- Streamlit web application
- Interactive UI components
- Real-time data updates
- Advanced visualization features
"""

from .app import BlackScholesApp, main

__all__ = [
    'BlackScholesApp',
    'main'
] 