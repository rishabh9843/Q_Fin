import streamlit as st
import pandas as pd
import numpy as np

# ==========================================
# IMPORT GUARD & LIBRARY
# ==========================================
try:
    from quant_library import (
        simulate_gbm, mc_option_price, bs_price, bs_greeks, 
        sma_crossover_strategy, backtest_metrics, mc_portfolio_var, DataHandler
    )
    import yfinance as yf
except ImportError as e:
    st.error("🚨 Library Import Failed!")
    st.write(f"Error Details: `{e}`")
    st.stop()

# ==========================================
# UI CONFIGURATION
# ==========================================
st.set_page_config(page_title="MFT Quant Workstation", layout="wide")

page = st.sidebar.radio("Select Module", 
    ["Market Simulation", "Option Pricing", "Live Backtester", "Portfolio VaR"])

# ==========================================
# MODULE 1: MARKET SIMULATION
# ==========================================
if page == "Market Simulation":
    st.header("Geometric Brownian Motion Simulator")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.subheader("Parameters")
        s0 = st.number_input("Initial Price", value=100.0, min_value=1.0)
        mu = st.slider("Drift (μ)", -0.2, 0.2, 0.05)
        sigma = st.slider("Volatility (σ)", 0.01, 1.0, 0.2)
        n_paths = st.slider("Number of Paths", 5, 100, 20) 
        t_days = st.number_input("Time Horizon (Days)", value=252, min_value=10)
    
    dt = 1/252
    T = t_days/252
    paths = simulate_gbm(s0, mu, sigma, T, dt, n_paths)
    
    with col2:
        if paths is not None:
            st.line_chart(paths)
            st.success("Simulation computed successfully!")

# ==========================================
# MODULE 2: DERIVATIVES
# ==========================================
elif page == "Option Pricing":
    st.header("Derivatives Analytics")
    
    c1, c2, c3, c4 = st.columns(4)
    S = c1.number_input("Stock Price", value=100.0, min_value=1.0)
    K = c2.number_input("Strike Price", value=105.0, min_value=1.0)
    T = c3.number_input("Expiry (Years)", value=1.0, min_value=0.01)
    sigma = c4.slider("Implied Vol", 0.01, 1.0, 0.25)
    
    r = 0.05 
    bs_val = bs_price(S, K, T, r, sigma)
    mc_val, se, _ = mc_option_price(S, K, T, r, sigma, n_sims=5000) 
    greeks = bs_greeks(S, K, T, r, sigma)
    
    st.subheader("Pricing Comparison")
    m1, m2, m3 = st.columns(3)
    m1.metric("Black-Scholes Price", f"${bs_val:.3f}")
    m2.metric("Monte Carlo Price", f"${mc_val:.3f}")
    m3.metric("Standard Error", f"{se:.4f}")
    
    st.write("### The Greeks")
    st.json(greeks)

# ==========================================
# MODULE 3: LIVE BACKTESTER (NEW!)
# ==========================================
elif page == "Live Backtester":
    st.header("Live Strategy Backtesting (Yahoo Finance)")
    
    ticker_input = st.sidebar.text_input("Enter Stock Ticker", value="AAPL").upper()
    short_w = st.sidebar.slider("SMA Short Window", 5, 50, 20)
    long_w = st.sidebar.slider("SMA Long Window", 51, 200, 100)
    
    st.write(f"Fetching 3 years of data for **{ticker_input}**...")
    
    try:
        raw_data = yf.download(ticker_input, period="3y", progress=False)
        
        if not raw_data.empty and 'Close' in raw_data.columns:
            # Extract just the Close series safely
            price_data = raw_data['Close'].squeeze()
            
            # Run the strategy
            results = sma_crossover_strategy(price_data, short_w, long_w)
            
            if len(results) > 0:
                metrics = backtest_metrics(results)
                
                # Display Metrics
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                col_m1.metric("Sharpe Ratio", f"{metrics['sharpe']:.2f}")
                col_m2.metric("Max Drawdown", f"{metrics['max_dd']*100:.2f}%")
                col_m3.metric("CAGR", f"{metrics['cagr']*100:.2f}%")
                col_m4.metric("Win Rate", f"{metrics['win_rate']*100:.1f}%")
                
                st.subheader("Equity Curve vs Buy & Hold")
                st.line_chart(results[['equity', 'bh_equity']])
            else:
                st.warning("Not enough data to calculate crossover. Reduce window sizes.")
        else:
            st.error(f"Could not fetch 'Close' data for {ticker_input}. Check the ticker symbol.")
            
    except Exception as e:
        st.error(f"An error occurred: {e}")

# ==========================================
# MODULE 4: PORTFOLIO RISK
# ==========================================
elif page == "Portfolio VaR":
    st.header("Market Risk Management (VaR)")
    
    S0_list = [100, 100, 100]
    weights = [0.4, 0.3, 0.3]
    mu_list = [0.05, 0.05, 0.05]
    sigma_list = [0.2, 0.25, 0.3]
    corr = np.array([[1.0, 0.6, 0.4], [0.6, 1.0, 0.5], [0.4, 0.5, 1.0]])
    
    var_results = mc_portfolio_var(S0_list, weights, mu_list, sigma_list, corr, T=1/252, n_paths=2000)
    
    st.write(f"**Current Portfolio Value:** ${var_results['portfolio_value_0']:.2f}")
    st.write(f"**95% VaR (1-Day):** ${var_results['var_95']:.2f}")
    st.write(f"**99% VaR (1-Day):** ${var_results['var_99']:.2f}")
    
    # Native Bar Chart for Distribution
    counts, bins = np.histogram(var_results['pnl'], bins=50)
    chart_data = pd.DataFrame({'Distribution Count': counts}, index=bins[:-1])
    st.bar_chart(chart_data)
