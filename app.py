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
def optimize_cvar(returns_np, target_return_period, alpha):
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

    mean_returns = returns_np.mean(axis=0) * returns_np.shape[0]
    constraints.append(w @ mean_returns >= target_return_period)

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
    plt.tight_layout()
    return fig


# -------------------------------
# Function 6: Plot cumulative returns
# -------------------------------
def plot_cum_returns(dates, cum_cvar, cum_equal):
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(dates, cum_cvar, label="CVaR-Optimized", color='blue', linewidth=2)
    ax.plot(dates, cum_equal, label="Equal-Weight", color='orange', linewidth=2)
    ax.set_title("Cumulative Returns Comparison")
    ax.set_ylabel("Cumulative Return")
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


# -------------------------------
# Function 7: Plot drawdowns
# -------------------------------
def plot_drawdowns(dates, drawdown_cvar, drawdown_equal):
    fig, ax = plt.subplots(figsize=(10,6))
    ax.fill_between(dates, drawdown_cvar * 100, 0, label="CVaR-Optimized", color='blue', alpha=0.3)
    ax.fill_between(dates, drawdown_equal * 100, 0, label="Equal-Weight", color='orange', alpha=0.3)
    ax.set_title("Portfolio Drawdowns")
    ax.set_ylabel("Drawdown (%)")
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
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
target_return_annual = st.slider("Target Annual Return (%)", 0, 25, 10, 1) / 100


if st.button("Run Optimization"):
    prices, returns = download_data(tickers, start_date, end_date)
    returns_np = returns.values
    
    # Convert annual target to period target
    years = (end_date - start_date).days / 365.25
    target_return_period = (1 + target_return_annual) ** years - 1

    # CVaR Optimization
    weights, cvar_value, expected_return_period = optimize_cvar(returns_np, target_return_period, alpha)
    
    if weights is None:
        st.error("Optimization failed. Try reducing the target return or adjusting other parameters.")
    else:
        # Calculate dynamic factors
        annualization_factor = np.sqrt(returns_np.shape[0] / years)
        
        # Convert back to annual for display
        expected_return_annual = (1 + expected_return_period) ** (1/years) - 1
        cvar_annual = (1 + cvar_value) ** (1/years) - 1

        # Display Results
        st.subheader("Optimization Results")
        st.write(f"**Status:** Optimal")
        st.write(f"**Expected Annual Return:** {expected_return_annual*100:.2f}%")
        st.write(f"**CVaR at {alpha*100:.0f}% confidence:** {cvar_annual*100:.2f}%")
        
        # Portfolio weights
        weights_df = pd.DataFrame({
            "Ticker": tickers, 
            "Weight (%)": weights * 100
        }).round(2)
        
        # Filter out near-zero weights for cleaner display
        weights_df = weights_df[weights_df["Weight (%)"] > 0.1].sort_values("Weight (%)", ascending=False)
        
        st.subheader("Portfolio Allocation")
        st.dataframe(weights_df, use_container_width=True)
        st.pyplot(plot_weights(weights_df["Ticker"], weights_df["Weight (%)"] / 100))

        # Compare with equal-weight portfolio
        equal_weights = np.ones(len(tickers)) / len(tickers)
        equal_return_period = np.mean(returns_np @ equal_weights) * returns_np.shape[0]
        equal_return_annual = (1 + equal_return_period) ** (1/years) - 1
        
        # Performance comparison
        st.subheader("Performance Comparison")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("CVaR-Optimized Portfolio", f"{expected_return_annual*100:.2f}%")
        with col2:
            st.metric("Equal-Weight Benchmark", f"{equal_return_annual*100:.2f}%", 
                     f"{(equal_return_annual - expected_return_annual)*100:.2f}%")

        # Cumulative returns
        cum_cvar = calculate_cum_returns(returns_np, weights)
        cum_equal = calculate_cum_returns(returns_np, equal_weights)

        st.subheader("Cumulative Returns")
        st.pyplot(plot_cum_returns(returns.index, cum_cvar, cum_equal))

        # Drawdowns
        drawdown_cvar = calculate_drawdown(cum_cvar)
        drawdown_equal = calculate_drawdown(cum_equal)

        st.subheader("Drawdown Analysis")
        st.pyplot(plot_drawdowns(returns.index, drawdown_cvar, drawdown_equal))
        
        # Summary statistics
        st.subheader("Risk Metrics")
        risk_metrics = pd.DataFrame({
            "Metric": ["Maximum Drawdown", f"CVaR ({alpha*100:.0f}%)", "Volatility (Annual)"],
            "CVaR Portfolio": [
                f"{drawdown_cvar.min()*100:.2f}%",
                f"{cvar_annual*100:.2f}%", 
                f"{np.std(returns_np @ weights) * annualization_factor * 100:.2f}%"
            ],
            "Equal Weight": [
                f"{drawdown_equal.min()*100:.2f}%",
                "N/A",
                f"{np.std(returns_np @ equal_weights) * annualization_factor * 100:.2f}%"
            ]
        })
        st.dataframe(risk_metrics, use_container_width=True)
        
        st.info("💡 **Interpretation:** CVaR-optimized portfolios focus on minimizing extreme losses. "
                "Lower drawdowns and CVaR values indicate better downside risk management.")