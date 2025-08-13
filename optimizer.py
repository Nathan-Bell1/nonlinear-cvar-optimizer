import cvxpy as cp
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.title("Mean-CVaR Portfolio Optimization")

# 1. User Select
tickers_input = st.text_input("Enter tickers separated by comma", 
                             "AAPL,MSFT,GOOGL,AMZN,TSLA,JPM,JNJ,V,PG,KO")
tickers = [t.strip().upper() for t in tickers_input.split(",")]


start_date = st.date_input("Start date", pd.to_datetime("2018-01-01"))
end_date = st.date_input("End date", pd.to_datetime("2023-01-01"))

alpha = st.slider("CVaR confidence level", 0.90, 0.99, 0.95)
target_return = st.slider("Minimum annual return (%)", 0.0, 0.20, 0.10) 
risk_free_rate = st.slider("Risk Free Rate:", 0, 0.05, 0.04,)

if st.button("Run Optimization")

    # 2. Download Data
    prices = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)[['Adj Close']]
    returns = prices.pct_change().dropna() 
    
    returns_np = returns.values  # Convert returns DataFrame to numpy array for speed
    n_scenarios, n_assets = returns_np.shape # Number of scenarios and assets

    # Step 3: CVaR Optimization
    w = cp.Variable(n_assets)      # Portfolio weights (what % in each asset)
    nu = cp.Variable()             # VaR estimate (Value at Risk threshold)
    xi = cp.Variable(n_scenarios, nonneg=True)   # Excess losses beyond VaR (for CVaR calc)

    portfolio_returns = returns_np @ w
    losses = -portfolio_returns

    objective = nu + (1/(1-alpha)) * cp.sum(xi) / n_scenarios

    constraints = [
        cp.sum(w) == 1,     # Fully invested
        w >= 0,             # No short selling
        xi >= losses - nu   # xi_i >= loss_i - nu
    ]

    mean_returns = returns_np.mean(axis=0) * n_scenarios # Annualized mean returns
    constraints.append(w @ mean_returns >= target_return) # Require at least 10% expected annual return

    # Setup and solve the problem
    prob = cp.Problem(cp.Minimize(objective), constraints)
    prob.solve(solver=cp.SCS)

    # Step 4: Display results
    st.subheader("Optimization Results")
    st.write(f"Status: {prob.status}")
    st.write(f"Optimal CVaR at {alpha*100:.0f}% confidence: {prob.value:.4f}")
    st.write(f"Expected annual return: {float(w.value @ mean_returns):.4f}")

    st.subheader("Portfolio Weights")
    weights_df = pd.DataFrame({"Ticker": returns.columns, "Weight": w.value})
    st.dataframe(weights_df)

    # Optional: plot weights
    fig, ax = plt.subplots()
    ax.bar(weights_df["Ticker"], weights_df["Weight"])
    ax.set_title("CVaR-Optimized Portfolio Weights")
    ax.set_ylabel("Weight")
    st.pyplot(fig)