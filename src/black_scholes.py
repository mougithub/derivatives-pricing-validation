"""Black-Scholes-Merton with continuous dividend yield q: price and analytic Greeks."""
import numpy as np
from scipy.stats import norm


def _check_kind(kind):
    if kind not in ("call", "put"):
        raise ValueError("kind must be 'call' or 'put'")


def d1_d2(S, K, T, r, sigma, q=0.0):
    srt = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / srt
    return d1, d1 - srt


def bs_price(S, K, T, r, sigma, q=0.0, kind="call"):
    """S may be an array; T and sigma must be scalars."""
    _check_kind(kind)
    S = np.asarray(S, dtype=float)
    if T <= 0 or sigma <= 0:  # limiting cases: discounted forward intrinsic
        fwd_pv = S * np.exp(-q * T) - K * np.exp(-r * T)
        return np.maximum(fwd_pv if kind == "call" else -fwd_pv, 0.0)
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    if kind == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def bs_greeks(S, K, T, r, sigma, q=0.0, kind="call"):
    """Delta, gamma, vega (per 1.00 of vol, not per 1%), theta (per year), rho (per 1.00 of rate)."""
    _check_kind(kind)
    S = np.asarray(S, dtype=float)
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    sq = np.sqrt(T)
    eq, er = np.exp(-q * T), np.exp(-r * T)
    pdf = norm.pdf(d1)
    gamma = eq * pdf / (S * sigma * sq)
    vega = S * eq * pdf * sq
    common = -S * eq * pdf * sigma / (2 * sq)
    if kind == "call":
        delta = eq * norm.cdf(d1)
        theta = common - r * K * er * norm.cdf(d2) + q * S * eq * norm.cdf(d1)
        rho = K * T * er * norm.cdf(d2)
    else:
        delta = -eq * norm.cdf(-d1)
        theta = common + r * K * er * norm.cdf(-d2) - q * S * eq * norm.cdf(-d1)
        rho = -K * T * er * norm.cdf(-d2)
    return dict(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)
