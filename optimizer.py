import cvxpy as cp
import numpy as np
import pandas as pd
import yfinance as yf

# Select 10 diverse tickers
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'JPM', 'JNJ', 'V', 'PG', 'KO']

# Download 5 years daily adjusted close price data
prices = yf.download(tickers, start='2018-01-01', end='2023-01-01', auto_adjust=False)[['Adj Close']]

returns = prices.pct_change().dropna() # Compute daily returns (simple returns)

# Convert returns DataFrame to numpy array for faster ops
returns_np = returns.values  # shape: (num_days, num_assets)


n_scenarios, n_assets = returns_np.shape # Number of scenarios and assets

alpha = 0.95  # Confidence level for CVaR
rf = 0.02     # Risk free rate 

# Variables
w = cp.Variable(n_assets)      # Portfolio weights (what % in each asset)
nu = cp.Variable()             # VaR estimate (Value at Risk threshold)
xi = cp.Variable(n_scenarios, nonneg=True)   # Excess losses beyond VaR (for CVaR calc)

# Calculate portfolio returns for each scenario: returns_np @ w is shape (n_scenarios,)
portfolio_returns = returns_np @ w
losses = -portfolio_returns

# CVaR objective
objective = nu + (1/(1-alpha)) * cp.sum(xi) / n_scenarios

constraints = [
    cp.sum(w) == 1,     # fully invested
    w >= 0,             # no short selling
    xi >= losses - nu   # xi_i >= loss_i - nu
]
