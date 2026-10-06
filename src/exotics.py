"""Closed-form benchmarks for exotics (continuous dividend yield q).

* down_and_out_call : continuously monitored, zero rebate (Reiner-Rubinstein form,
                      notation as in Haug, "Complete Guide to Option Pricing Formulas").
* geometric_asian_price : discretely monitored with n_fix equally spaced fixings
                      t_i = iT/n (exact); n_fix=None gives the continuous
                      (Kemna-Vorst) limit.
* digital_call_price : cash-or-nothing call.
Known-bad cases (naive MC barrier, pathwise digital delta, ...) are added on Day 3.
"""
import numpy as np
from scipy.stats import norm
from black_scholes import bs_price

N = norm.cdf


def down_and_out_call(S, K, H, T, r, sigma, q=0.0):
    if H >= S:
        return 0.0                      # already knocked out
    if H <= 0:
        return float(bs_price(S, K, T, r, sigma, q, "call"))
    srt = sigma * np.sqrt(T)
    lam = (r - q + 0.5 * sigma**2) / sigma**2   # = 1 + mu in Haug's notation
    dq, dr = np.exp(-q * T), np.exp(-r * T)
    hs = H / S
    if H <= K:                          # price = vanilla - down-and-in
        y = np.log(H * H / (S * K)) / srt + lam * srt
        c_di = S * dq * hs ** (2 * lam) * N(y) - K * dr * hs ** (2 * lam - 2) * N(y - srt)
        return float(bs_price(S, K, T, r, sigma, q, "call") - c_di)
    x2 = np.log(S / H) / srt + lam * srt            # H > K : B - D
    y2 = np.log(H / S) / srt + lam * srt
    B = S * dq * N(x2) - K * dr * N(x2 - srt)
    D = S * dq * hs ** (2 * lam) * N(y2) - K * dr * hs ** (2 * lam - 2) * N(y2 - srt)
    return float(B - D)


def down_and_in_call(S, K, H, T, r, sigma, q=0.0):
    return float(bs_price(S, K, T, r, sigma, q, "call")) - down_and_out_call(S, K, H, T, r, sigma, q)


def _geo_moments(T, n_fix):
    if n_fix is None:
        return 0.5, T / 3.0
    n = float(n_fix)
    return (n + 1) / (2 * n), T * (n + 1) * (2 * n + 1) / (6 * n * n)


def geometric_asian_price(S, K, T, r, sigma, q=0.0, n_fix=None, kind="call"):
    if kind not in ("call", "put"):
        raise ValueError("kind must be 'call' or 'put'")
    a, v = _geo_moments(T, n_fix)
    mu = np.log(S) + (r - q - 0.5 * sigma**2) * T * a     # mean of ln G
    var = sigma**2 * v                                    # variance of ln G
    sd = np.sqrt(var)
    EG = np.exp(mu + 0.5 * var)
    d2 = (mu - np.log(K)) / sd
    d1 = d2 + sd
    disc = np.exp(-r * T)
    if kind == "call":
        return float(disc * (EG * N(d1) - K * N(d2)))
    return float(disc * (K * N(-d2) - EG * N(-d1)))


def geometric_asian_forward(S, T, r, sigma, q=0.0, n_fix=None):
    """E[G] under the risk-neutral measure (used for put-call parity)."""
    a, v = _geo_moments(T, n_fix)
    return float(np.exp(np.log(S) + (r - q - 0.5 * sigma**2) * T * a + 0.5 * sigma**2 * v))


def digital_call_price(S, K, T, r, sigma, q=0.0, payout=1.0):
    srt = sigma * np.sqrt(T)
    d2 = (np.log(S / K) + (r - q - 0.5 * sigma**2) * T) / srt
    return float(payout * np.exp(-r * T) * N(d2))
