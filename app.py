import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.title("Mean-CVaR Portfolio Optimization")


# -------------------------------
# Function 1: Download and process data
# -------------------------------
def download_data(tickers, start, end):
    prices = yf.download(tickers, start=start, end=end, auto_adjust=False)[['Adj Close']]
    returns = prices.pct_change().dropna()
    return prices, returns


# -------------------------------
# Function 2: CVaR optimization
# -------------------------------
def optimize_cvar(returns_np, target_return, alpha):
    n_scenarios, n_assets = returns_np.shape

    w = cp.Variable(n_assets)
    nu = cp.Variable()
    xi = cp.Variable(n_scenarios, nonneg=True)

    portfolio_returns = returns_np @ w
    losses = -portfolio_returns

    objective = nu + (1/(1-alpha)) * cp.sum(xi) / n_scenarios

    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        xi >= losses - nu
    ]

    mean_returns = returns_np.mean(axis=0) * 252  # Annualized
    constraints.append(w @ mean_returns >= target_return)

    prob = cp.Problem(cp.Minimize(objective), constraints)
    prob.solve(solver=cp.SCS)

    return w.value, prob.value, float(w.value @ mean_returns)


# -------------------------------
# Function 3: Cumulative returns
# -------------------------------
def calculate_cum_returns(returns_np, weights):
    return (returns_np @ weights + 1).cumprod()


# -------------------------------
# Function 4: Drawdown
# -------------------------------
def calculate_drawdown(cum_returns):
    peak = np.maximum.accumulate(cum_returns)
    drawdown = (cum_returns - peak) / peak
    return drawdown


# -------------------------------
# Function 5: Plot weights
# -------------------------------
def plot_weights(tickers, weights):
    fig, ax = plt.subplots(figsize=(8,5))
    ax.bar(tickers, weights, color='skyblue')
    ax.set_title("CVaR-Optimized Portfolio Weights")
    ax.set_ylabel("Weight")
    plt.xticks(rotation=45)
    return fig


# -------------------------------
# Function 6: Plot cumulative returns
# -------------------------------
def plot_cum_returns(dates, cum_cvar, cum_equal):
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(dates, cum_cvar, label="CVaR-Optimized", color='blue')
    ax.plot(dates, cum_equal, label="Equal-Weight", color='orange')
    ax.set_title("Cumulative Returns")
    ax.set_ylabel("Cumulative Return")
    ax.set_xlabel("Date")
    ax.legend()
    return fig


# -------------------------------
# Function 7: Plot drawdowns
# -------------------------------
def plot_drawdowns(dates, drawdown_cvar, drawdown_equal):
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(dates, drawdown_cvar, label="CVaR-Optimized", color='blue')
    ax.plot(dates, drawdown_equal, label="Equal-Weight", color='orange')
    ax.set_title("Portfolio Drawdowns")
    ax.set_ylabel("Drawdown (%)")
    ax.set_xlabel("Date")
    ax.legend()
    return fig


# -------------------------------
# Streamlit User Inputs
# -------------------------------
tickers_input = st.text_input(
    "Enter tickers separated by comma", 
    "AAPL,MSFT,GOOGL,AMZN,TSLA,JPM,JNJ,V,PG,KO"
)
tickers = [t.strip().upper() for t in tickers_input.split(",")]

start_date = st.date_input("Start Date", pd.to_datetime("2018-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("2023-01-01"))

alpha = st.slider("CVaR Confidence Level (%)", 85.0, 99.0, 95.0, 0.1) / 100
target_return = st.slider("Minimum Annual Return (%)", 0, 20, 10, 1) / 100


if st.button("Run Optimization"):
    prices, returns = download_data(tickers, start_date, end_date)
    returns_np = returns.values

    # CVaR Optimization
    weights, cvar_value, expected_return = optimize_cvar(returns_np, target_return, alpha)

    # Display Results
    st.subheader("Optimization Results")
    st.write(f"Status: {'Optimal' if weights is not None else 'Failed'}")
    st.write(f"Optimal CVaR at {alpha*100:.0f}% confidence: {cvar_value:.4f}")
    st.write(f"Expected annual return: {expected_return:.4f}")

    weights_df = pd.DataFrame({"Ticker": tickers, "Weight": np.abs(weights) * 100})
    st.dataframe(weights_df.style.format({"Weight": lambda x: "-" if x < 0.01 else f"{x:.2f}%"}))
    st.pyplot(plot_weights(tickers, weights))

    # Compare with equal-weight portfolio
    equal_weights = np.ones(len(tickers)) / len(tickers)
    cum_cvar = calculate_cum_returns(returns_np, weights)
    cum_equal = calculate_cum_returns(returns_np, equal_weights)

    st.subheader("Cumulative Returns Comparison")
    st.pyplot(plot_cum_returns(returns.index, cum_cvar, cum_equal))

    # Drawdowns
    drawdown_cvar = calculate_drawdown(cum_cvar)
    drawdown_equal = calculate_drawdown(cum_equal)

    st.subheader("Drawdowns")
    st.pyplot(plot_drawdowns(returns.index, drawdown_cvar, drawdown_equal))
    st.write("Drawdown closer to 0 = smaller losses; CVaR-optimized portfolio should have smaller extreme drops compared to equal-weight.")
