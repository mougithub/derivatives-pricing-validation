"""Greeks four ways for the vanilla call (and digital delta): analytic | bump-and-revalue with common
random numbers | pathwise | PDE grid.  adds the delta-gamma vs full-repricing test to this module.

Estimators return Est(value, se). Bump estimators use ONE set of normals for every bumped price (CRN).
Pathwise gamma is not provided for the call: the integrand max(S_T-K,0) has zero second derivative almost
everywhere, so the naive pathwise estimator would silently return 0.
"""
from dataclasses import dataclass
import numpy as np
from black_scholes import bs_price, bs_greeks
from exotics import down_and_out_call
from pde_pricer import pde_price


@dataclass
class Est:
    value: float
    se: float

    def within(self, ref, z, bias_rel):
        """|value - ref| <= z*SE + bias_rel*|ref|  (statistical error plus a finite-difference/smoothing allowance)."""
        return abs(self.value - ref) <= z * self.se + bias_rel * abs(ref)


def _est(samples):
    return Est(float(samples.mean()), float(samples.std(ddof=1) / np.sqrt(samples.size)))


def _terminal(S0, T, r, sigma, q, z):
    return S0 * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)


def mc_call_greeks_bump(S, K, T, r, sigma, q=0.0, n_paths=1_000_000, h=None, h_rel=0.01,
                        vol_bump=0.005, seed=0):
    """Central bump-and-revalue with common random numbers. h (absolute) overrides h_rel*S."""
    h = h_rel * S if h is None else h
    z = np.random.default_rng(seed).standard_normal(n_paths)
    disc = np.exp(-r * T)
    pay = lambda S0, sig: disc * np.maximum(_terminal(S0, T, r, sig, q, z) - K, 0.0)
    up, mid, dn = pay(S + h, sigma), pay(S, sigma), pay(S - h, sigma)
    return dict(delta=_est((up - dn) / (2 * h)),
                gamma=_est((up - 2 * mid + dn) / h**2),
                vega=_est((pay(S, sigma + vol_bump) - pay(S, sigma - vol_bump)) / (2 * vol_bump)))


def mc_call_greeks_pathwise(S, K, T, r, sigma, q=0.0, n_paths=1_000_000, seed=0):
    """Pathwise delta and vega: differentiate the integrand along each path."""
    z = np.random.default_rng(seed).standard_normal(n_paths)
    ST = _terminal(S, T, r, sigma, q, z)
    itm = (ST > K).astype(float)
    disc = np.exp(-r * T)
    delta = disc * itm * ST / S
    vega = disc * itm * ST * (np.log(ST / S) - (r - q + 0.5 * sigma**2) * T) / sigma
    return dict(delta=_est(delta), vega=_est(vega))


def pde_vanilla_greeks(S, K, T, r, sigma, q=0.0, kind="call", M=800, N=800, n_sd=6.0,
                       vol_bump=0.005, rannacher=False):
    """Delta, gamma from grid differences; vega by central bump with the grid held FIXED."""
    half = n_sd * sigma * np.sqrt(T)
    kw = dict(M=M, N=N, rannacher=rannacher, half_width=half)
    base = pde_price(kind, S, K, T, r, sigma, q, **kw)
    up = pde_price(kind, S, K, T, r, sigma + vol_bump, q, **kw)["price"]
    dn = pde_price(kind, S, K, T, r, sigma - vol_bump, q, **kw)["price"]
    return dict(price=base["price"], delta=base["delta"], gamma=base["gamma"], vega=(up - dn) / (2 * vol_bump))


# ---------------- digital (cash-or-nothing call, payout 1): delta ----------------
def mc_digital_delta_bump(S, K, T, r, sigma, q=0.0, n_paths=1_000_000, h=None, h_rel=0.01, seed=0):
    h = h_rel * S if h is None else h
    z = np.random.default_rng(seed).standard_normal(n_paths)
    disc = np.exp(-r * T)
    ind = lambda S0: (_terminal(S0, T, r, sigma, q, z) > K).astype(float)
    return _est(disc * (ind(S + h) - ind(S - h)) / (2 * h))


def mc_digital_delta_pathwise_naive(S, K, T, r, sigma, q=0.0, n_paths=1_000_000, seed=0):
    """KNOWN-BAD. Pathwise differentiates the integrand 1{S_T > K}: it is piecewise constant, so its
    derivative w.r.t. S0 is 0 on every path (almost surely). The estimator returns exactly 0 with SE 0:
    a confident, silent, wrong answer."""
    z = np.random.default_rng(seed).standard_normal(n_paths)
    _ = _terminal(S, T, r, sigma, q, z)            # paths are simulated, but the derivative of the indicator is 0
    return _est(np.zeros(n_paths))


def mc_digital_delta_pathwise_smoothed(S, K, T, r, sigma, q=0.0, n_paths=1_000_000, eps=None, eps_rel=0.01, seed=0):
    """Fix: replace 1{S_T>K} by a linear ramp from K-eps to K+eps (a call spread / 2eps), then go pathwise:
    delta samples = e^{-rT} * 1{|S_T-K|<eps} / (2 eps) * S_T/S0.  Bias O(eps^2), variance O(1/eps)."""
    eps = eps_rel * S if eps is None else eps
    z = np.random.default_rng(seed).standard_normal(n_paths)
    ST = _terminal(S, T, r, sigma, q, z)
    return _est(np.exp(-r * T) * (np.abs(ST - K) < eps) / (2 * eps) * ST / S)


# =======================  delta-gamma(-vega) approximation vs full repricing =======================

@dataclass(frozen=True)
class Position:
    kind: str          # "call" | "put" | "doc" (down-and-out call, continuous barrier, zero rebate)
    K: float
    qty: float         # signed: -1 = short
    H: float = None


def position_value(pos, S, T, r, sigma, q):
    if pos.kind in ("call", "put"):
        return pos.qty * float(bs_price(S, pos.K, T, r, sigma, q, pos.kind))
    if pos.kind == "doc":
        return pos.qty * down_and_out_call(S, pos.K, pos.H, T, r, sigma, q)   # 0 once S <= H (barrier already hit)
    raise ValueError(pos.kind)


def book_value(book, S, T, r, sigma, q):
    return sum(position_value(p, S, T, r, sigma, q) for p in book)


def gross_value(book, S, T, r, sigma, q):
    return sum(abs(position_value(p, S, T, r, sigma, q)) for p in book)


def book_greeks(book, S, T, r, sigma, q, h_rel=1e-3, h_vol=1e-4):
    """Book delta, gamma, vega at the base point. Vanilla legs: analytic. Barrier leg: central differences of the
    closed-form price (valid while the base point is several bump-widths from the barrier)."""
    tot = dict(delta=0.0, gamma=0.0, vega=0.0)
    h = h_rel * S
    for p in book:
        if p.kind in ("call", "put"):
            g = bs_greeks(S, p.K, T, r, sigma, q, p.kind)
            for k in tot:
                tot[k] += p.qty * float(g[k])
        else:
            f = lambda s, sg: position_value(p, s, T, r, sg, q)
            tot["delta"] += (f(S + h, sigma) - f(S - h, sigma)) / (2 * h)
            tot["gamma"] += (f(S + h, sigma) - 2 * f(S, sigma) + f(S - h, sigma)) / h**2
            tot["vega"] += (f(S, sigma + h_vol) - f(S, sigma - h_vol)) / (2 * h_vol)
    return tot


def approx_pnl(g, dS, dsigma=0.0, use_vega=True):
    """Delta-gamma (-vega) P&L approximation for an instantaneous shock."""
    return g["delta"] * dS + 0.5 * g["gamma"] * dS**2 + (g["vega"] * dsigma if use_vega else 0.0)


def full_pnl(book, S, T, r, sigma, q, dS, dsigma=0.0):
    return book_value(book, S + dS, T, r, sigma + dsigma, q) - book_value(book, S, T, r, sigma, q)


def flag_metric(approx, full, gross, floor_frac):
    """|approx - full| / max(|full|, floor_frac * gross).  'Unreliable' if above the flag threshold."""
    return abs(approx - full) / max(abs(full), floor_frac * gross)


def reliability_frontier(book, S, T, r, sigma, q, floor_frac, flag_rel, step, max_s, use_vega=False):
    """Smallest |s| (grid of `step`) at which delta-gamma is flagged, separately for down and up shocks.
    Returns (down, up); max_s + step if never flagged up to max_s."""
    g = book_greeks(book, S, T, r, sigma, q)
    gross = gross_value(book, S, T, r, sigma, q)
    out = []
    for sign in (-1, +1):
        found = max_s + step
        for k in range(1, int(round(max_s / step)) + 1):
            dS = sign * k * step * S
            if flag_metric(approx_pnl(g, dS, 0.0, use_vega), full_pnl(book, S, T, r, sigma, q, dS), gross, floor_frac) > flag_rel:
                found = k * step
                break
        out.append(found)
    return tuple(out)
