"""
Core quantitative finance functions.
Monte Carlo, Black-Scholes, Greeks, GBM, VaR, Backtesting.
"""

import numpy as np
from scipy.stats import norm
import pandas as pd


# ─── DATA HANDLER (For Real Market Data) ──────────────────────────────────────

class DataHandler:
    def __init__(self, ticker: str):
        self.ticker = ticker.upper()
        self.data = None
        self.S = None
        self.sigma = None

    def get_stock_data(self) -> float:
        try:
            df = yf.download(self.ticker, period="5d", progress=False)
            if df.empty:
                raise ValueError(f"No data found for the ticker: {self.ticker}")
            if 'Close' in df.columns:
                close_prices = df['Close'].squeeze()
                self.S = float(close_prices.iloc[-1])
                return self.S
            raise ValueError("'Close' data is not available.")
        except Exception as e:
            print(f"Error fetching stock data: {e}")
            return None

    def calculate_historical_volatility(self, start_date: str, end_date: str, window: int = 252) -> float:
        try:
            df = yf.download(self.ticker, start=start_date, end=end_date, progress=False)
            if df.empty:
                raise ValueError(f"No data found for the ticker: {self.ticker}")
            if 'Close' in df.columns:
                self.data = df
                close_prices = df['Close'].squeeze()
                returns = np.log(close_prices / close_prices.shift(1))
                self.sigma = float(returns.std() * np.sqrt(window))
                return self.sigma
            raise ValueError("'Close' data is not available.")
        except Exception as e:
            print(f"Error fetching historical volatility: {e}")
            return None

# ─── GBM SIMULATION ───────────────────────────────────────────────────────────

def simulate_gbm(S0: float, mu: float, sigma: float, T: float,
                 dt: float, n_paths: int, seed: int = 42) -> np.ndarray:
    sigma = max(sigma, 1e-4) 
    T = max(T, 1e-4)
    dt = max(dt, 1e-4)

    np.random.seed(seed)
    n_steps = int(T / dt)
    if n_steps <= 0: n_steps = 1
        
    Z = np.random.standard_normal((n_steps, n_paths))
    log_returns = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
    price_paths = np.zeros((n_steps + 1, n_paths))
    price_paths[0] = S0
    for t in range(1, n_steps + 1):
        price_paths[t] = price_paths[t - 1] * np.exp(log_returns[t - 1])
    return price_paths

def simulate_correlated_gbm(S0_list, mu_list, sigma_list, corr_matrix, T, dt, n_paths, seed=42):
    np.random.seed(seed)
    n_assets = len(S0_list)
    T = max(T, 1e-4)
    dt = max(dt, 1e-4)
    n_steps = int(T / dt)
    if n_steps <= 0: n_steps = 1
        
    sigma_list = [max(s, 1e-4) for s in sigma_list]
    L = np.linalg.cholesky(corr_matrix)
    
    paths = {i: np.zeros((n_steps + 1, n_paths)) for i in range(n_assets)}
    for i in range(n_assets): paths[i][0] = S0_list[i]

    for t in range(1, n_steps + 1):
        Z_ind = np.random.standard_normal((n_assets, n_paths))
        Z_corr = L @ Z_ind
        for i in range(n_assets):
            paths[i][t] = paths[i][t - 1] * np.exp(
                (mu_list[i] - 0.5 * sigma_list[i] ** 2) * dt + sigma_list[i] * np.sqrt(dt) * Z_corr[i]
            )
    return paths

# ─── BLACK-SCHOLES & GREEKS ───────────────────────────────────────────────────

def d1(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0: return 0.0
    return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

def d2(S, K, T, r, sigma):
    return d1(S, K, T, r, sigma) - sigma * np.sqrt(T)

def bs_price(S, K, T, r, sigma, option_type="call"):
    if T <= 0: return max(0.0, S - K) if option_type == "call" else max(0.0, K - S)
    _d1 = d1(S, K, T, r, sigma)
    _d2 = d2(S, K, T, r, sigma)
    if option_type == "call": return S * norm.cdf(_d1) - K * np.exp(-r * T) * norm.cdf(_d2)
    return K * np.exp(-r * T) * norm.cdf(-_d2) - S * norm.cdf(-_d1)

def bs_greeks(S, K, T, r, sigma, option_type="call"):
    if T <= 0 or sigma <= 0: return {"delta": 0, "gamma": 0, "vega": 0, "theta": 0, "rho": 0}
    _d1 = d1(S, K, T, r, sigma)
    _d2 = d2(S, K, T, r, sigma)
    pdf_d1 = norm.pdf(_d1)
    sign = 1 if option_type == "call" else -1

    delta = sign * norm.cdf(sign * _d1)
    gamma = pdf_d1 / (S * sigma * np.sqrt(T))
    vega  = S * pdf_d1 * np.sqrt(T) / 100
    theta = (-(S * pdf_d1 * sigma) / (2 * np.sqrt(T)) - sign * r * K * np.exp(-r * T) * norm.cdf(sign * _d2)) / 365
    rho   = sign * K * T * np.exp(-r * T) * norm.cdf(sign * _d2) / 100
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}

def mc_option_price(S, K, T, r, sigma, option_type="call", n_sims=10000, seed=42, antithetic=True):
    np.random.seed(seed)
    T, sigma = max(T, 1e-4), max(sigma, 1e-4)
    
    if antithetic:
        half = n_sims // 2
        Z = np.random.standard_normal(half)
        Z = np.concatenate([Z, -Z])
    else: Z = np.random.standard_normal(n_sims)

    ST = S * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)
    payoffs = np.maximum(0, ST - K) if option_type == "call" else np.maximum(0, K - ST)

    discount = np.exp(-r * T)
    price = discount * np.mean(payoffs)
    std_err = discount * np.std(payoffs) / np.sqrt(n_sims)
    return price, std_err, ST

# ─── RISK & BACKTESTING ───────────────────────────────────────────────────────

def mc_portfolio_var(S0_list, weights, mu_list, sigma_list, corr_matrix, T, n_paths=5000, seed=42):
    paths = simulate_correlated_gbm(S0_list, mu_list, sigma_list, corr_matrix, T, T, n_paths, seed)
    portfolio_value_0 = sum(w * s for w, s in zip(weights, S0_list))
    portfolio_value_T = sum(weights[i] * paths[i][1] for i in range(len(S0_list)))
    
    pnl = portfolio_value_T - portfolio_value_0
    pnl_sorted = np.sort(pnl)

    var_95, var_99 = -np.percentile(pnl, 5), -np.percentile(pnl, 1)
    return {"pnl": pnl, "var_95": var_95, "var_99": var_99, "portfolio_value_0": portfolio_value_0}

def cagr(equity_curve: pd.Series, periods: int = 252) -> float:
    n_years = len(equity_curve) / periods
    if n_years <= 0 or len(equity_curve) < 2: return 0.0
    try: return float((equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / n_years) - 1)
    except: return 0.0

def max_drawdown(equity_curve: pd.Series) -> float:
    if len(equity_curve) == 0: return 0.0
    roll_max = equity_curve.cummax()
    drawdown = (equity_curve - roll_max) / roll_max
    return float(drawdown.min())

def sharpe_ratio(returns: pd.Series, rf: float = 0.065, periods: int = 252) -> float:
    excess = returns - rf / periods
    if returns.std() == 0 or len(returns) == 0: return 0.0
    return float(np.sqrt(periods) * excess.mean() / returns.std())

def sma_crossover_strategy(prices: pd.Series, short_window: int = 20, long_window: int = 50) -> pd.DataFrame:
    df = pd.DataFrame({"price": prices})
    df["sma_short"] = prices.rolling(short_window).mean()
    df["sma_long"]  = prices.rolling(long_window).mean()
    df["signal"]    = 0
    df.loc[df["sma_short"] > df["sma_long"], "signal"] = 1
    df.loc[df["sma_short"] < df["sma_long"], "signal"] = -1
    df["position"]  = df["signal"].shift(1).fillna(0)
    df["returns"]   = prices.pct_change()
    df["strategy_returns"] = df["position"] * df["returns"]
    df["equity"]    = (1 + df["strategy_returns"].fillna(0)).cumprod()
    df["bh_equity"] = (1 + df["returns"].fillna(0)).cumprod()
    return df.dropna()

def backtest_metrics(df: pd.DataFrame) -> dict:
    if "strategy_returns" not in df.columns or len(df) == 0:
        return {"sharpe": 0, "max_dd": 0, "cagr": 0, "win_rate": 0}
    r, eq = df["strategy_returns"].dropna(), df["equity"].dropna()
    return {"sharpe": sharpe_ratio(r), "max_dd": max_drawdown(eq), "cagr": cagr(eq), 
            "win_rate": float((r > 0).sum() / max(len(r[r != 0]), 1))}