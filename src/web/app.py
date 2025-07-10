"""
Main Streamlit web application for the Black-Scholes Options Pricer.

This module provides a comprehensive web interface for:
- Interactive option pricing
- Real-time market data
- Greeks analysis
- Risk assessment
- Strategy comparison
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any
import plotly.graph_objects as go

# Use absolute imports instead of relative imports
from models.black_scholes import BlackScholesModel, OptionParameters
from models.greeks import GreeksCalculator
from utils.data_fetcher import MarketDataFetcher
from utils.validators import InputValidator
from visualization.charts import OptionCharts


class BlackScholesApp:
    """Main Streamlit application for Black-Scholes option pricing"""
    
    def __init__(self):
        self.data_fetcher = MarketDataFetcher()
        self.setup_page()
    
    def setup_page(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title="Interactive Black-Scholes Options Pricer",
            page_icon="📈",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("📈 Interactive Black-Scholes Options Pricer")
        st.markdown("---")
    
    def sidebar_inputs(self) -> Dict[str, Any]:
        """Create sidebar input widgets"""
        st.sidebar.header("Option Parameters")
        
        # Stock price input
        S = st.sidebar.number_input(
            "Current Stock Price ($)",
            min_value=0.01,
            max_value=10000.0,
            value=100.0,
            step=0.01,
            format="%.2f"
        )
        
        # Strike price input
        K = st.sidebar.number_input(
            "Strike Price ($)",
            min_value=0.01,
            max_value=10000.0,
            value=100.0,
            step=0.01,
            format="%.2f"
        )
        
        # Time to expiration input
        T = st.sidebar.number_input(
            "Time to Expiration (years)",
            min_value=0.01,
            max_value=10.0,
            value=1.0,
            step=0.01,
            format="%.2f"
        )
        
        # Risk-free rate input
        r = st.sidebar.number_input(
            "Risk-free Rate (%)",
            min_value=-10.0,
            max_value=50.0,
            value=3.0,
            step=0.1,
            format="%.1f"
        ) / 100  # Convert to decimal
        
        # Volatility input
        sigma = st.sidebar.number_input(
            "Volatility (%)",
            min_value=0.1,
            max_value=200.0,
            value=20.0,
            step=0.1,
            format="%.1f"
        ) / 100  # Convert to decimal
        
        # Option type selection
        option_type = st.sidebar.selectbox(
            "Option Type",
            ["call", "put"],
            format_func=lambda x: x.title()
        )
        
        return {
            "S": S, "K": K, "T": T, "r": r, 
            "sigma": sigma, "option_type": option_type
        }
    
    def market_data_section(self):
        """Market data fetching section"""
        st.header("📊 Market Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            symbol = st.text_input("Stock Symbol", value="AAPL").upper()
            if st.button("Fetch Market Data"):
                try:
                    # Fetch stock data
                    stock_data = self.data_fetcher.get_stock_data(symbol)
                    current_price = stock_data['Close'].iloc[-1]
                    
                    # Calculate historical volatility
                    hist_vol = self.data_fetcher.calculate_historical_volatility(symbol)
                    
                    # Get risk-free rate
                    rf_rate = self.data_fetcher.get_risk_free_rate()
                    
                    st.success(f"✅ Data fetched successfully!")
                    st.metric("Current Price", f"${current_price:.2f}")
                    st.metric("Historical Volatility", f"{hist_vol:.1%}")
                    st.metric("Risk-free Rate", f"{rf_rate:.1%}")
                    
                    # Update sidebar with fetched data
                    st.session_state.update({
                        "S": current_price,
                        "sigma": hist_vol,
                        "r": rf_rate
                    })
                    
                except Exception as e:
                    st.error(f"❌ Error fetching data: {str(e)}")
        
        with col2:
            st.subheader("Historical Price Chart")
            if 'stock_data' in locals():
                # Create price chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=stock_data.index,
                    y=stock_data['Close'],
                    mode='lines',
                    name='Stock Price'
                ))
                fig.update_layout(
                    title=f"{symbol} Historical Price",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def pricing_results(self, params: Dict[str, Any]):
        """Display option pricing results"""
        st.header("💰 Option Pricing Results")
        
        # Create option parameters
        option_params = OptionParameters(**params)
        
        # Calculate option price
        model = BlackScholesModel(option_params)
        price = model.price()
        
        # Calculate Greeks
        greeks_calc = GreeksCalculator(model)
        greeks = greeks_calc.all_greeks()
        
        # Display results in columns
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
            st.metric("Moneyness", 
                     "ITM" if model.is_in_the_money() else "OTM")
        
        # Option details table
        st.subheader("Option Details")
        details_df = pd.DataFrame({
            "Parameter": ["Stock Price", "Strike Price", "Time to Expiry", 
                         "Risk-free Rate", "Volatility", "Option Type"],
            "Value": [f"${params['S']:.2f}", f"${params['K']:.2f}", 
                     f"{params['T']:.2f} years", f"{params['r']:.1%}", 
                     f"{params['sigma']:.1%}", params['option_type'].title()]
        })
        st.table(details_df)
    
    def sensitivity_analysis(self, params: Dict[str, Any]):
        """Sensitivity analysis section"""
        st.header("📊 Sensitivity Analysis")
        
        option_params = OptionParameters(**params)
        model = BlackScholesModel(option_params)
        
        # Create tabs for different analyses
        tab1, tab2, tab3 = st.tabs(["Price Sensitivity", "Greeks Analysis", "P&L Chart"])
        
        with tab1:
            st.subheader("Option Price Sensitivity")
            
            # Stock price sensitivity
            stock_range = np.linspace(params['S'] * 0.7, params['S'] * 1.3, 50)
            fig1 = OptionCharts.price_sensitivity_chart(
                model, "S", stock_range, 
                "Option Price vs Stock Price"
            )
            st.plotly_chart(fig1, use_container_width=True)
            
            # Volatility sensitivity
            vol_range = np.linspace(0.05, 0.5, 50)
            fig2 = OptionCharts.price_sensitivity_chart(
                model, "sigma", vol_range,
                "Option Price vs Volatility"
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        with tab2:
            st.subheader("Greeks Analysis")
            
            stock_prices = np.linspace(params['S'] * 0.7, params['S'] * 1.3, 100)
            fig = OptionCharts.greeks_chart(model, stock_prices)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Profit/Loss at Expiration")
            
            fig = OptionCharts.profit_loss_chart(
                params['K'], model.price(), params['option_type']
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def risk_analysis(self, params: Dict[str, Any]):
        """Risk analysis section"""
        st.header("⚠️ Risk Analysis")
        
        option_params = OptionParameters(**params)
        model = BlackScholesModel(option_params)
        greeks_calc = GreeksCalculator(model)
        
        # Calculate risk metrics
        greeks = greeks_calc.all_greeks()
        risk_metrics = greeks_calc.risk_metrics()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Risk Metrics")
            
            # Display risk metrics
            st.metric("Delta Exposure", f"{risk_metrics['delta_exposure']:.4f}")
            st.metric("Gamma Exposure", f"{risk_metrics['gamma_exposure']:.4f}")
            st.metric("Vega Exposure", f"{risk_metrics['vega_exposure']:.4f}")
            st.metric("Theta Decay", f"{risk_metrics['theta_decay']:.4f}")
            st.metric("Total Risk", f"{risk_metrics['total_risk']:.4f}")
        
        with col2:
            st.subheader("Probability Analysis")
            
            # Calculate probabilities
            prob_profit = model.calculate_probability_of_profit()
            expected_value = model.calculate_expected_value()
            
            st.metric("Probability of Profit", f"{prob_profit:.1%}")
            st.metric("Expected Value", f"${expected_value:.4f}")
            st.metric("Option Status", model.get_option_status())
            st.metric("Moneyness", f"{model.moneyness():.2f}")
        
        # Risk metrics chart
        st.subheader("Risk Metrics Overview")
        fig = OptionCharts.risk_metrics_chart(risk_metrics)
        st.plotly_chart(fig, use_container_width=True)
    
    def advanced_features(self, params: Dict[str, Any]):
        """Advanced features section"""
        st.header("🚀 Advanced Features")
        
        tab1, tab2, tab3 = st.tabs(["Implied Volatility", "Strategy Builder", "Monte Carlo"])
        
        with tab1:
            st.subheader("Implied Volatility Calculator")
            
            # Input for market price
            market_price = st.number_input(
                "Market Option Price ($)",
                min_value=0.01,
                value=float(BlackScholesModel(OptionParameters(**params)).price()),
                step=0.01,
                format="%.4f"
            )
            
            if st.button("Calculate Implied Volatility"):
                try:
                    from models.volatility import VolatilityCalculator
                    
                    implied_vol = VolatilityCalculator.implied_volatility(
                        market_price, params['S'], params['K'], 
                        params['T'], params['r'], params['option_type']
                    )
                    
                    st.success(f"Implied Volatility: {implied_vol:.1%}")
                    
                    # Compare with input volatility
                    diff = abs(implied_vol - params['sigma'])
                    st.info(f"Difference from input: {diff:.1%}")
                    
                except Exception as e:
                    st.error(f"Error calculating implied volatility: {str(e)}")
        
        with tab2:
            st.subheader("Option Strategy Builder")
            
            # Simple strategy comparison
            strategies = st.multiselect(
                "Select Strategies to Compare",
                ["Long Call", "Long Put", "Covered Call", "Protective Put"],
                default=["Long Call", "Long Put"]
            )
            
            if st.button("Compare Strategies"):
                # Create strategy comparison chart
                options_data = []
                stock_prices = np.linspace(params['S'] * 0.7, params['S'] * 1.3, 100)
                
                for strategy in strategies:
                    if strategy == "Long Call":
                        payoffs = np.maximum(stock_prices - params['K'], 0) - model.price()
                    elif strategy == "Long Put":
                        payoffs = np.maximum(params['K'] - stock_prices, 0) - model.price()
                    else:
                        payoffs = np.zeros_like(stock_prices)
                    
                    options_data.append({
                        'name': strategy,
                        'stock_prices': stock_prices,
                        'payoffs': payoffs
                    })
                
                fig = OptionCharts.option_comparison_chart(options_data)
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Monte Carlo Simulation")
            
            n_paths = st.slider("Number of Paths", 100, 10000, 1000)
            n_steps = st.slider("Number of Steps", 50, 500, 100)
            
            if st.button("Run Monte Carlo Simulation"):
                try:
                    # Simple Monte Carlo simulation
                    np.random.seed(42)
                    
                    # Generate stock price paths
                    dt = params['T'] / n_steps
                    paths = np.zeros((n_paths, n_steps + 1))
                    paths[:, 0] = params['S']
                    
                    for i in range(n_steps):
                        z = np.random.standard_normal(n_paths)
                        paths[:, i + 1] = paths[:, i] * np.exp(
                            (params['r'] - 0.5 * params['sigma']**2) * dt + 
                            params['sigma'] * np.sqrt(dt) * z
                        )
                    
                    # Calculate option payoffs
                    if params['option_type'] == "call":
                        payoffs = np.maximum(paths[:, -1] - params['K'], 0)
                    else:
                        payoffs = np.maximum(params['K'] - paths[:, -1], 0)
                    
                    # Discount to present value
                    mc_price = np.exp(-params['r'] * params['T']) * np.mean(payoffs)
                    mc_error = np.exp(-params['r'] * params['T']) * np.std(payoffs) / np.sqrt(n_paths)
                    
                    st.metric("Monte Carlo Price", f"${mc_price:.4f}")
                    st.metric("Standard Error", f"${mc_error:.4f}")
                    
                    # Create simulation chart
                    fig = go.Figure()
                    for i in range(min(100, n_paths)):  # Show first 100 paths
                        fig.add_trace(go.Scatter(
                            x=np.linspace(0, params['T'], n_steps + 1),
                            y=paths[i, :],
                            mode='lines',
                            line=dict(color='blue', opacity=0.1),
                            showlegend=False
                        ))
                    
                    fig.update_layout(
                        title="Monte Carlo Stock Price Paths",
                        xaxis_title="Time (years)",
                        yaxis_title="Stock Price ($)",
                        template="plotly_white"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error running Monte Carlo simulation: {str(e)}")
    
    def run(self):
        """Main application runner"""
        # Sidebar inputs
        params = self.sidebar_inputs()
        
        # Main content
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Market Data", "Pricing Results", "Sensitivity Analysis", 
            "Risk Analysis", "Advanced Features"
        ])
        
        with tab1:
            self.market_data_section()
        
        with tab2:
            self.pricing_results(params)
        
        with tab3:
            self.sensitivity_analysis(params)
        
        with tab4:
            self.risk_analysis(params)
        
        with tab5:
            self.advanced_features(params)


def main():
    """Main function to run the Streamlit app"""
    app = BlackScholesApp()
    app.run()


if __name__ == "__main__":
    main() 