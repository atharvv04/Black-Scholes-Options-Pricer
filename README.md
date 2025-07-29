# Interactive Black-Scholes Options Pricer

A comprehensive, interactive web application for pricing European options using the Black-Scholes model with real-time market data integration.

## Features

- **Real-time Market Data**: Fetch current stock prices and calculate historical volatility
- **Interactive Pricing**: Calculate option prices with live parameter updates
- **Greeks Analysis**: Complete Greeks calculation (Delta, Gamma, Theta, Vega, Rho)
- **Sensitivity Analysis**: Visualize how option prices change with parameter variations
- **Profit/Loss Charts**: Interactive P&L charts at expiration
- **Risk Analysis**: Comprehensive risk metrics and probability calculations
- **Implied Volatility**: Calculate implied volatility from market prices
- **Monte Carlo Simulation**: Advanced pricing with simulation
- **Strategy Builder**: Compare different option strategies

## Installation

```bash
# Clone the repository
git clone https://github.com/atharvv04/Black-Scholes-Options-Pricer.git
cd Black-Scholes-Options-Pricer

# Install dependencies
pip install -r requirements.txt

# Test the installation
python test_imports.py

# Run the application (recommended)
python launch_app.py

# Alternative methods
python run_app.py
# or
streamlit run simple_app.py
```

## Deployment

### Local Development
```bash
# Run locally
python launch_app.py
```

### Streamlit Cloud Deployment
1. **Deploy to GitHub**: Run the deployment script
   ```bash
   python deploy_to_github.py
   ```

2. **Deploy to Streamlit Cloud**:
   - Go to [Streamlit Cloud](https://share.streamlit.io/)
   - Connect the GitHub repository
   - Set the main file to: `streamlit_app.py`
   - Deploy

3. **The app will be live at**: `https://your-app-name.streamlit.app`

## Usage

1. **Market Data**: Enter a stock symbol to fetch current market data
2. **Parameters**: Adjust option parameters in the sidebar
3. **Results**: View pricing results, Greeks, and sensitivity analysis
4. **Analysis**: Explore different scenarios and strategies

## Mathematical Foundation

The Black-Scholes model prices European options using:

**Call Option**: C = S*N(d₁) - K*e^(-rT)*N(d₂)
**Put Option**: P = K*e^(-rT)*N(-d₂) - S*N(-d₁)

Where:
- d₁ = (ln(S/K) + (r + σ²/2)T) / (σ√T)
- d₂ = d₁ - σ√T
- S = Current stock price
- K = Strike price
- T = Time to expiration
- r = Risk-free rate
- σ = Volatility
- N() = Cumulative normal distribution

## Project Structure

```
black-scholes-pricer/
├── src/                    # Source code
│   ├── models/            # Core pricing models
│   │   ├── black_scholes.py
│   │   ├── greeks.py
│   │   └── volatility.py
│   ├── utils/             # Utilities and helpers
│   │   ├── data_fetcher.py
│   │   ├── validators.py
│   │   └── calculators.py
│   ├── visualization/     # Charting and plotting
│   │   └── charts.py
│   └── web/              # Web application
│       └── app.py
├── tests/                 # Unit tests
├── docs/                  # Documentation
└── examples/              # Usage examples
```

## Core Components

### Black-Scholes Model
- Complete implementation of the Black-Scholes formula
- Parameter validation and error handling
- Intrinsic value and time value calculations
- Moneyness and option status analysis

### Greeks Calculator
- Delta: Price sensitivity to the underlying asset price
- Gamma: Delta sensitivity to the underlying asset price
- Theta: Price sensitivity to time decay
- Vega: Price sensitivity to volatility
- Rho: Price sensitivity to interest rate

### Market Data Integration
- Real-time stock price fetching via yfinance
- Historical volatility calculation
- Risk-free rate retrieval
- Options chain data access

### Visualization
- Interactive Plotly charts
- Price sensitivity analysis
- Greeks visualization
- Profit/Loss charts
- Risk metrics radar charts

## API Examples

### Basic Option Pricing

```python
from src.models.black_scholes import BlackScholesModel, OptionParameters

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
print(f"Call option price: ${price:.4f}")
```

### Greeks Analysis

```python
from src.models.greeks import GreeksCalculator

# Calculate all Greeks
greeks_calc = GreeksCalculator(model)
greeks = greeks_calc.all_greeks()

print(f"Delta: {greeks['delta']:.4f}")
print(f"Gamma: {greeks['gamma']:.4f}")
print(f"Theta: {greeks['theta']:.4f}")
print(f"Vega: {greeks['vega']:.4f}")
print(f"Rho: {greeks['rho']:.4f}")
```

### Market Data Integration

```python
from src.utils.data_fetcher import MarketDataFetcher

# Fetch market data
fetcher = MarketDataFetcher()
current_price = fetcher.get_current_price("AAPL")
hist_vol = fetcher.calculate_historical_volatility("AAPL")
rf_rate = fetcher.get_risk_free_rate()

print(f"Current Price: ${current_price:.2f}")
print(f"Historical Volatility: {hist_vol:.1%}")
print(f"Risk-free Rate: {rf_rate:.1%}")
```

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_black_scholes.py

# Run with coverage
pytest --cov=src tests/
```
