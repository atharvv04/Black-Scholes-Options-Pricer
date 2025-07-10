#!/usr/bin/env python3
"""
Streamlit Cloud deployment file for Black-Scholes Options Pricer.

This is the main entry point for Streamlit Cloud deployment.
"""

import streamlit as st
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import our modules
from models.black_scholes import BlackScholesModel, OptionParameters
from models.greeks import GreeksCalculator
from utils.data_fetcher import MarketDataFetcher
from visualization.charts import OptionCharts

def main():
    """Main Streamlit application for deployment"""
    st.set_page_config(
        page_title="Black-Scholes Options Pricer",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📈 Black-Scholes Options Pricer")
    st.markdown("---")
    
    # Sidebar inputs
    st.sidebar.header("Option Parameters")
    
    S = st.sidebar.number_input(
        "Current Stock Price ($)",
        min_value=0.01,
        max_value=10000.0,
        value=100.0,
        step=0.01,
        format="%.2f"
    )
    
    K = st.sidebar.number_input(
        "Strike Price ($)",
        min_value=0.01,
        max_value=10000.0,
        value=100.0,
        step=0.01,
        format="%.2f"
    )
    
    T = st.sidebar.number_input(
        "Time to Expiration (years)",
        min_value=0.01,
        max_value=10.0,
        value=1.0,
        step=0.01,
        format="%.2f"
    )
    
    r = st.sidebar.number_input(
        "Risk-free Rate (%)",
        min_value=-10.0,
        max_value=50.0,
        value=3.0,
        step=0.1,
        format="%.1f"
    ) / 100
    
    sigma = st.sidebar.number_input(
        "Volatility (%)",
        min_value=0.1,
        max_value=200.0,
        value=20.0,
        step=0.1,
        format="%.1f"
    ) / 100
    
    option_type = st.sidebar.selectbox(
        "Option Type",
        ["call", "put"],
        format_func=lambda x: x.title()
    )
    
    # Create option parameters
    params = OptionParameters(
        S=S, K=K, T=T, r=r, sigma=sigma, option_type=option_type
    )
    
    # Calculate option price
    model = BlackScholesModel(params)
    price = model.price()
    
    # Calculate Greeks
    greeks_calc = GreeksCalculator(model)
    greeks = greeks_calc.all_greeks()
    
    # Display results
    st.header("💰 Option Pricing Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Option Price", f"${price:.4f}")
        st.metric("Intrinsic Value", f"${model.intrinsic_value():.4f}")
        st.metric("Time Value", f"${model.time_value():.4f}")
    
    with col2:
        st.metric("Delta", f"{greeks['delta']:.4f}")
        st.metric("Gamma", f"{greeks['gamma']:.4f}")
        st.metric("Theta", f"{greeks['theta']:.4f}")
    
    with col3:
        st.metric("Vega", f"{greeks['vega']:.4f}")
        st.metric("Rho", f"{greeks['rho']:.4f}")
        st.metric("Status", model.get_option_status())
    
    # Option details table
    st.subheader("Option Details")
    import pandas as pd
    details_df = pd.DataFrame({
        "Parameter": ["Stock Price", "Strike Price", "Time to Expiry", 
                     "Risk-free Rate", "Volatility", "Option Type"],
        "Value": [f"${S:.2f}", f"${K:.2f}", f"{T:.2f} years", 
                 f"{r:.1%}", f"{sigma:.1%}", option_type.title()]
    })
    st.table(details_df)
    
    # Sensitivity analysis
    st.header("📊 Sensitivity Analysis")
    
    # Stock price sensitivity
    import numpy as np
    stock_range = np.linspace(S * 0.7, S * 1.3, 50)
    fig1 = OptionCharts.price_sensitivity_chart(
        model, "S", stock_range, "Option Price vs Stock Price"
    )
    st.plotly_chart(fig1, use_container_width=True)
    
    # Greeks chart
    stock_prices = np.linspace(S * 0.7, S * 1.3, 100)
    fig2 = OptionCharts.greeks_chart(model, stock_prices)
    st.plotly_chart(fig2, use_container_width=True)
    
    # Profit/Loss chart
    fig3 = OptionCharts.profit_loss_chart(K, price, option_type)
    st.plotly_chart(fig3, use_container_width=True)
    
    # Market data section
    st.header("📊 Market Data")
    
    symbol = st.text_input("Stock Symbol", value="AAPL").upper()
    if st.button("Fetch Market Data"):
        try:
            fetcher = MarketDataFetcher()
            current_price = fetcher.get_current_price(symbol)
            hist_vol = fetcher.calculate_historical_volatility(symbol)
            rf_rate = fetcher.get_risk_free_rate()
            
            st.success(f"✅ Data fetched for {symbol}!")
            st.metric("Current Price", f"${current_price:.2f}")
            st.metric("Historical Volatility", f"{hist_vol:.1%}")
            st.metric("Risk-free Rate", f"{rf_rate:.1%}")
            
        except Exception as e:
            st.error(f"❌ Error fetching data: {str(e)}")

if __name__ == "__main__":
    main() 