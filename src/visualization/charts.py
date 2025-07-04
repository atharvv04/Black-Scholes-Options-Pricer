"""
Interactive charts for option analysis.

This module provides comprehensive charting capabilities for:
- Option price sensitivity analysis
- Greeks visualization
- Profit/Loss charts
- Risk analysis plots
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List, Tuple, Optional
import pandas as pd

from models.black_scholes import BlackScholesModel, OptionParameters
from models.greeks import GreeksCalculator


class OptionCharts:
    """Create interactive charts for option analysis"""
    
    @staticmethod
    def price_sensitivity_chart(model: BlackScholesModel, 
                              parameter: str, 
                              range_values: List[float],
                              title: str = "") -> go.Figure:
        """
        Create price sensitivity chart
        
        Args:
            model: BlackScholesModel instance
            parameter: Parameter to vary ("S", "K", "T", "r", "sigma")
            range_values: Values to test
            title: Chart title
        
        Returns:
            Plotly figure
        """
        prices = []
        for value in range_values:
            # Create new parameters with varied value
            params_dict = model.params.__dict__.copy()
            params_dict[parameter] = value
            new_params = OptionParameters(**params_dict)
            new_model = BlackScholesModel(new_params)
            prices.append(new_model.price())
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=range_values,
            y=prices,
            mode='lines+markers',
            name=f'Option Price vs {parameter.upper()}',
            line=dict(color='blue', width=2),
            marker=dict(size=6)
        ))
        
        # Add current value marker
        current_value = getattr(model.params, parameter)
        current_price = model.price()
        fig.add_trace(go.Scatter(
            x=[current_value],
            y=[current_price],
            mode='markers',
            name='Current Value',
            marker=dict(color='red', size=10, symbol='diamond')
        ))
        
        fig.update_layout(
            title=title or f"Option Price Sensitivity to {parameter.upper()}",
            xaxis_title=parameter.upper(),
            yaxis_title="Option Price ($)",
            template="plotly_white",
            hovermode='x unified'
        )
        
        return fig
    
    @staticmethod
    def greeks_chart(model: BlackScholesModel, 
                    stock_prices: List[float]) -> go.Figure:
        """
        Create Greeks chart
        
        Args:
            model: BlackScholesModel instance
            stock_prices: Range of stock prices
        
        Returns:
            Plotly figure with subplots for each Greek
        """
        deltas, gammas, thetas, vegas, rhos = [], [], [], [], []
        
        for S in stock_prices:
            params_dict = model.params.__dict__.copy()
            params_dict['S'] = S
            new_params = OptionParameters(**params_dict)
            new_model = BlackScholesModel(new_params)
            greeks = GreeksCalculator(new_model).all_greeks()
            
            deltas.append(greeks['delta'])
            gammas.append(greeks['gamma'])
            thetas.append(greeks['theta'])
            vegas.append(greeks['vega'])
            rhos.append(greeks['rho'])
        
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Delta', 'Gamma', 'Theta', 'Vega', 'Rho'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Add traces
        fig.add_trace(go.Scatter(x=stock_prices, y=deltas, name="Delta", 
                                line=dict(color='blue')), row=1, col=1)
        fig.add_trace(go.Scatter(x=stock_prices, y=gammas, name="Gamma", 
                                line=dict(color='red')), row=1, col=2)
        fig.add_trace(go.Scatter(x=stock_prices, y=thetas, name="Theta", 
                                line=dict(color='green')), row=2, col=1)
        fig.add_trace(go.Scatter(x=stock_prices, y=vegas, name="Vega", 
                                line=dict(color='orange')), row=2, col=2)
        fig.add_trace(go.Scatter(x=stock_prices, y=rhos, name="Rho", 
                                line=dict(color='purple')), row=3, col=1)
        
        # Add current stock price marker
        current_S = model.params.S
        current_greeks = GreeksCalculator(model).all_greeks()
        
        fig.add_trace(go.Scatter(
            x=[current_S], y=[current_greeks['delta']], 
            mode='markers', name='Current Delta',
            marker=dict(color='blue', size=8, symbol='diamond')
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=[current_S], y=[current_greeks['gamma']], 
            mode='markers', name='Current Gamma',
            marker=dict(color='red', size=8, symbol='diamond')
        ), row=1, col=2)
        
        fig.update_layout(
            title="Option Greeks vs Stock Price",
            height=600,
            template="plotly_white",
            showlegend=True
        )
        
        return fig
    
    @staticmethod
    def profit_loss_chart(K: float, premium: float, option_type: str,
                         stock_range: Optional[Tuple[float, float]] = None,
                         title: str = "") -> go.Figure:
        """
        Create profit/loss chart
        
        Args:
            K: Strike price
            premium: Option premium
            option_type: "call" or "put"
            stock_range: Tuple of (min_stock, max_stock)
            title: Chart title
        
        Returns:
            Plotly figure
        """
        if stock_range is None:
            if option_type == "call":
                stock_range = (K * 0.7, K * 1.3)
            else:
                stock_range = (K * 0.7, K * 1.3)
        
        stock_prices = np.linspace(stock_range[0], stock_range[1], 100)
        
        # Calculate payoffs
        if option_type == "call":
            payoffs = np.maximum(stock_prices - K, 0) - premium
        else:  # put option
            payoffs = np.maximum(K - stock_prices, 0) - premium
        
        fig = go.Figure()
        
        # Add P&L line
        color = 'green' if option_type == "call" else 'red'
        fig.add_trace(go.Scatter(
            x=stock_prices,
            y=payoffs,
            mode='lines',
            name=f'{option_type.title()} P&L',
            line=dict(color=color, width=3)
        ))
        
        # Add breakeven line
        if option_type == "call":
            breakeven = K + premium
        else:
            breakeven = K - premium
        
        fig.add_vline(x=breakeven, line_dash="dash", line_color="gray",
                     annotation_text="Breakeven")
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="black")
        
        # Add strike price line
        fig.add_vline(x=K, line_dash="dot", line_color="blue",
                     annotation_text="Strike Price")
        
        fig.update_layout(
            title=title or f"{option_type.title()} Option Profit/Loss at Expiration",
            xaxis_title="Stock Price at Expiration ($)",
            yaxis_title="Profit/Loss ($)",
            template="plotly_white",
            hovermode='x unified'
        )
        
        return fig
    
    @staticmethod
    def volatility_smile_chart(strikes: List[float], 
                              implied_vols: List[float],
                              title: str = "") -> go.Figure:
        """
        Create volatility smile chart
        
        Args:
            strikes: List of strike prices
            implied_vols: List of implied volatilities
            title: Chart title
        
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=strikes,
            y=implied_vols,
            mode='lines+markers',
            name='Implied Volatility',
            line=dict(color='blue', width=2),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title=title or "Volatility Smile",
            xaxis_title="Strike Price ($)",
            yaxis_title="Implied Volatility",
            template="plotly_white"
        )
        
        return fig
    
    @staticmethod
    def term_structure_chart(expirations: List[float], 
                           implied_vols: List[float],
                           title: str = "") -> go.Figure:
        """
        Create volatility term structure chart
        
        Args:
            expirations: List of expiration times
            implied_vols: List of implied volatilities
            title: Chart title
        
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=expirations,
            y=implied_vols,
            mode='lines+markers',
            name='Implied Volatility',
            line=dict(color='red', width=2),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title=title or "Volatility Term Structure",
            xaxis_title="Time to Expiration (years)",
            yaxis_title="Implied Volatility",
            template="plotly_white"
        )
        
        return fig
    
    @staticmethod
    def risk_metrics_chart(risk_metrics: Dict[str, float],
                          title: str = "") -> go.Figure:
        """
        Create risk metrics radar chart
        
        Args:
            risk_metrics: Dictionary of risk metrics
            title: Chart title
        
        Returns:
            Plotly figure
        """
        # Prepare data for radar chart
        categories = list(risk_metrics.keys())
        values = list(risk_metrics.values())
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Risk Metrics',
            line_color='blue'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(values) * 1.2]
                )),
            showlegend=False,
            title=title or "Risk Metrics Overview",
            template="plotly_white"
        )
        
        return fig
    
    @staticmethod
    def option_comparison_chart(options_data: List[Dict],
                               title: str = "") -> go.Figure:
        """
        Create option comparison chart
        
        Args:
            options_data: List of option data dictionaries
            title: Chart title
        
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
        
        for i, option in enumerate(options_data):
            color = colors[i % len(colors)]
            
            fig.add_trace(go.Scatter(
                x=option['stock_prices'],
                y=option['payoffs'],
                mode='lines',
                name=option['name'],
                line=dict(color=color, width=2)
            ))
        
        fig.add_hline(y=0, line_dash="dash", line_color="black")
        
        fig.update_layout(
            title=title or "Option Strategy Comparison",
            xaxis_title="Stock Price at Expiration ($)",
            yaxis_title="Profit/Loss ($)",
            template="plotly_white"
        )
        
        return fig
    
    @staticmethod
    def monte_carlo_simulation_chart(simulation_paths: np.ndarray,
                                   time_steps: np.ndarray,
                                   title: str = "") -> go.Figure:
        """
        Create Monte Carlo simulation chart
        
        Args:
            simulation_paths: Array of simulation paths
            time_steps: Time steps array
            title: Chart title
        
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Plot sample paths (first 100)
        for i in range(min(100, simulation_paths.shape[0])):
            fig.add_trace(go.Scatter(
                x=time_steps,
                y=simulation_paths[i, :],
                mode='lines',
                line=dict(color='lightblue', width=1),
                showlegend=False
            ))
        
        # Plot mean path
        mean_path = np.mean(simulation_paths, axis=0)
        fig.add_trace(go.Scatter(
            x=time_steps,
            y=mean_path,
            mode='lines',
            line=dict(color='red', width=3),
            name='Mean Path'
        ))
        
        fig.update_layout(
            title=title or "Monte Carlo Stock Price Simulation",
            xaxis_title="Time (years)",
            yaxis_title="Stock Price ($)",
            template="plotly_white"
        )
        
        return fig 