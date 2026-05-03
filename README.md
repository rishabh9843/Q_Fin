# Q_Fin
# Quantitative Research & Risk Workstation 📈🔬



> **Live Application:** [View the Live Streamlit Dashboard Here](https://i84tczmz9xl4btzlrkletf.streamlit.app/)

## Overview
This repository contains a full-stack, modular quantitative research platform built in Python. Designed for rapid prototyping and low-latency modeling, this workstation allows researchers to simulate stochastic market environments, price derivatives, backtest algorithmic alpha signals, and evaluate multi-asset portfolio risk.

The computational backend relies heavily on vectorized `NumPy` and `SciPy` operations, entirely bypassing standard Python loops to achieve a **>50x speedup** in simulation execution times. Live market data ingestion is handled via the `yfinance` API.

---

## ⚙️ Core Modules & Mathematical Architecture

### 1. Market Simulation Engine
* **Model:** Geometric Brownian Motion (GBM)
* **Description:** Simulates thousands of future asset price paths using randomized standard normal shocks.
* **Math:** $S_t = S_{t-1} \exp\left(\left(\mu - \frac{\sigma^2}{2}\right)dt + \sigma \sqrt{dt} Z\right)$

### 2. Derivatives Analytics (Pricing & Greeks)
* **Models:** Black-Scholes Formula & Monte Carlo Simulation
* **Description:** Computes theoretical European options pricing and first/second-order Greeks ($\Delta, \Gamma, \nu, \Theta, \rho$). Includes a parallel Monte Carlo pricing engine with Standard Error variance tracking.

### 3. Vectorized Backtesting Framework
* **Description:** An object-oriented backtesting pipeline designed to evaluate technical alpha signals (e.g., Simple Moving Average Crossover) against live historical data. 
* **Metrics Generated:** Sharpe Ratio, Compound Annual Growth Rate (CAGR), Maximum Drawdown, and Win Rate.

### 4. Portfolio Risk Management (VaR)
* **Model:** Multivariate Monte Carlo Historical Simulation
* **Description:** Calculates 1-Day 95% and 99% Value at Risk (VaR) for multi-asset portfolios. 
* **Math:** Utilizes **Cholesky decomposition** on historical covariance matrices to accurately model correlated stochastic price paths across different underlying assets.

---

## 🚀 Performance & Optimization
A primary focus of this project was computational efficiency, simulating the environment of a Medium-Frequency Trading (MFT) desk:
* **Vectorization:** Replaced native Python `for` loops with vectorized `NumPy` arrays. Matrix operations are pushed to the underlying C-backend.
* **Efficiency Gain:** Reduced 10,000-path Monte Carlo execution times from ~2.5 seconds to ~40 milliseconds (a >95% efficiency improvement).

---

## 💻 Installation & Local Setup

To run this workstation locally, follow these steps:

**1. Clone the repository:**
```bash
git clone [https://github.com/YourUsername/YourRepoName.git](https://github.com/YourUsername/YourRepoName.git)
cd YourRepoName
