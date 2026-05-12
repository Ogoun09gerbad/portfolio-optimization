import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.portfolio import compute_returns, compute_covariance
from src.optimizer import optimize_portfolio

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Portfolio Optimizer", layout="wide")

st.title("📊 Portfolio Optimization Dashboard")
st.markdown("Optimize your investment portfolio using Modern Portfolio Theory")

# =========================
# USER INPUT
# =========================
stocks = st.multiselect(
    "Select assets",
    ["AAPL", "MSFT", "TSLA", "NVDA", "GOOGL", "AMZN"],
    default=["AAPL", "MSFT", "TSLA"]
)

start_date = st.date_input("Start date", value=pd.to_datetime("2020-01-01"))

# =========================
# DATA LOADING
# =========================
if len(stocks) > 1:

    data = yf.download(stocks, start=start_date)["Close"]
    data = data.dropna()

    st.subheader("📈 Asset Prices")
    st.line_chart(data)

    # =========================
    # RETURNS
    # =========================
    returns = compute_returns(data)

    st.subheader("📉 Returns")
    st.write(returns.tail())

    # =========================
    # COVARIANCE
    # =========================
    cov_matrix = compute_covariance(returns)

    st.subheader("📊 Covariance Matrix")
    st.write(cov_matrix)

    # =========================
    # OPTIMIZATION
    # =========================
    weights, sharpe = optimize_portfolio(returns)

    st.subheader("🚀 Optimal Portfolio")

    st.write(f"Sharpe Ratio: {sharpe:.4f}")

    weights_series = pd.Series(weights, index=stocks)
    st.write(weights_series)

    # =========================
    # VISUALIZATION
    # =========================
    fig, ax = plt.subplots()
    ax.pie(weights, labels=stocks, autopct="%1.1f%%")
    ax.set_title("Optimal Allocation")
    st.pyplot(fig)

else:
    st.warning("Please select at least 2 assets")
