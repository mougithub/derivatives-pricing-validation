import itertools
import numpy as np
import pytest
import tolerances as T
from black_scholes import bs_price, bs_greeks

GRID = list(itertools.product([50.0, 80.0, 100.0, 120.0, 200.0],   # S
                              [0.1, 1.0, 5.0],                      # T
                              [0.0, 0.05],                          # r
                              [0.0, 0.03],                          # q
                              [0.1, 0.2, 0.5]))                     # sigma
K = 100.0


def test_put_call_parity():
    worst = 0.0
    for S, t, r, q, s in GRID:
        c = bs_price(S, K, t, r, s, q, "call")
        p = bs_price(S, K, t, r, s, q, "put")
        worst = max(worst, abs(c - p - (S * np.exp(-q * t) - K * np.exp(-r * t))))
    assert worst < T.PARITY_ABS, worst


@pytest.mark.parametrize("kind", ["call", "put"])
def test_price_bounds(kind):
    for S, t, r, q, s in GRID:
        v = float(bs_price(S, K, t, r, s, q, kind))
        fwd = S * np.exp(-q * t) - K * np.exp(-r * t)
        lower = max(fwd if kind == "call" else -fwd, 0.0)
        upper = S * np.exp(-q * t) if kind == "call" else K * np.exp(-r * t)
        assert lower - T.BOUNDS_SLACK <= v <= upper + T.BOUNDS_SLACK, (S, t, r, q, s, v)


def test_monotonicity():
    spots = np.linspace(40, 200, 400)
    c = bs_price(spots, K, 1.0, 0.05, 0.2, 0.02, "call")
    p = bs_price(spots, K, 1.0, 0.05, 0.2, 0.02, "put")
    assert np.all(np.diff(c) > 0) and np.all(np.diff(p) < 0)
    for kind in ("call", "put"):
        vols = np.linspace(0.05, 1.0, 200)
        v = np.array([bs_price(100.0, K, 1.0, 0.05, s, 0.02, kind) for s in vols])
        assert np.all(np.diff(v) > 0)


def test_limit_zero_vol_and_zero_maturity():
    for S in (80.0, 100.0, 120.0):
        for kind in ("call", "put"):
            fwd = S * np.exp(-0.02) - K * np.exp(-0.05)
            ref = max(fwd if kind == "call" else -fwd, 0.0)
            assert abs(bs_price(S, K, 1.0, 0.05, 1e-6, 0.02, kind) - ref) < T.LIMIT_ABS
            intr = max(S - K, 0.0) if kind == "call" else max(K - S, 0.0)
            assert abs(bs_price(S, K, 1e-10, 0.05, 0.2, 0.02, kind) - intr) < T.LIMIT_ABS


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("S", [80.0, 100.0, 120.0])
def test_greeks_vs_central_difference(S, kind):
    args = dict(K=K, T=1.0, r=0.05, q=0.02)
    f = lambda s, sig: float(bs_price(s, args["K"], args["T"], args["r"], sig, args["q"], kind))
    g = bs_greeks(S, K, 1.0, 0.05, 0.2, 0.02, kind)
    h, hv = 0.01, 1e-4
    delta = (f(S + h, 0.2) - f(S - h, 0.2)) / (2 * h)
    h2 = 0.1
    gamma = (f(S + h2, 0.2) - 2 * f(S, 0.2) + f(S - h2, 0.2)) / h2**2
    vega = (f(S, 0.2 + hv) - f(S, 0.2 - hv)) / (2 * hv)
    for name, fd in (("delta", delta), ("gamma", gamma), ("vega", vega)):
        rel = abs(float(g[name]) - fd) / abs(fd)
        assert rel < T.GREEKS_FD_REL, (name, S, kind, rel)
