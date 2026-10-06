import numpy as np
import pytest
import tolerances as T
from black_scholes import bs_price
from exotics import (down_and_out_call, down_and_in_call, geometric_asian_price,
                     geometric_asian_forward, digital_call_price)

R = T.REF


def _do(K, H, **kw):
    p = dict(S=R["S"], T=R["T"], r=R["r"], sigma=R["sigma"], q=R["q"]); p.update(kw)
    return down_and_out_call(p["S"], K, H, p["T"], p["r"], p["sigma"], p["q"])


# ---------- barrier ----------
def test_barrier_branch_continuity_at_H_equals_K():
    for K in (85.0, 90.0, 95.0):
        eps = 1e-12   # gap must be << tolerance/|dP/dH| (slope ~ -0.7)
        below = _do(K, K - eps)   # H<=K branch
        above = _do(K, K + eps)   # H>K branch
        assert abs(below - above) < T.BARRIER_CONTINUITY_ABS, (K, below, above)


def test_barrier_vanilla_limit_and_zero_limit():
    for K in (90.0, 100.0, 110.0):
        v = float(bs_price(R["S"], K, R["T"], R["r"], R["sigma"], R["q"], "call"))
        assert abs(_do(K, 1e-6) - v) < T.BARRIER_VANILLA_LIMIT_ABS
        assert abs(_do(K, R["S"] - 1e-7)) < T.BARRIER_NEAR_S_ABS


def test_barrier_bounds_and_monotone_in_H():
    K = 90.0
    v = float(bs_price(R["S"], K, R["T"], R["r"], R["sigma"], R["q"], "call"))
    Hs = np.linspace(40, 99, 60)            # crosses K=90, so both branches are used
    prices = np.array([_do(K, H) for H in Hs])
    assert np.all(prices >= -1e-12) and np.all(prices <= v + 1e-12)
    assert np.all(np.diff(prices) <= 1e-12)   # higher barrier -> lower knock-out value
    di = np.array([down_and_in_call(R["S"], K, H, R["T"], R["r"], R["sigma"], R["q"]) for H in Hs])
    assert np.allclose(prices + di, v, atol=1e-12)


def _mc_barrier_bgk(K, H, seed):
    """Independent check: fine discrete monitoring + Broadie-Glasserman-Kou shift.
    Lives in the test on purpose: Day 3 builds the real, documented version."""
    n, steps = T.BARRIER_MC_N_PATHS, T.BARRIER_MC_N_STEPS
    dt = R["T"] / steps
    # discrete MC approximating a CONTINUOUS down barrier: shift barrier TOWARD spot
    Hadj = H * np.exp(+0.5826 * R["sigma"] * np.sqrt(dt))
    rng = np.random.default_rng(seed)
    S = np.full(n, R["S"]); alive = np.ones(n, bool)
    drift = (R["r"] - R["q"] - 0.5 * R["sigma"]**2) * dt
    vol = R["sigma"] * np.sqrt(dt)
    for _ in range(steps):
        S *= np.exp(drift + vol * rng.standard_normal(n))
        alive &= S > Hadj
    pay = np.exp(-R["r"] * R["T"]) * np.where(alive, np.maximum(S - K, 0.0), 0.0)
    return pay.mean(), pay.std(ddof=1) / np.sqrt(n)


@pytest.mark.parametrize("K,H", [(100.0, 90.0), (90.0, 95.0)])   # H<=K branch, H>K branch
def test_barrier_closed_form_vs_independent_mc(K, H):
    est, se = _mc_barrier_bgk(K, H, seed=[T.SEED, 3, int(K), int(H)])
    cf = _do(K, H)
    assert abs(est - cf) / se < T.BARRIER_MC_Z_MAX, (est, cf, se)


# ---------- geometric Asian ----------
@pytest.mark.parametrize("n_fix", [1, 4, 12, 252, None])
def test_geometric_asian_put_call_parity(n_fix):
    for K in (80.0, 100.0, 120.0):
        c = geometric_asian_price(R["S"], K, R["T"], R["r"], R["sigma"], R["q"], n_fix, "call")
        p = geometric_asian_price(R["S"], K, R["T"], R["r"], R["sigma"], R["q"], n_fix, "put")
        eg = geometric_asian_forward(R["S"], R["T"], R["r"], R["sigma"], R["q"], n_fix)
        assert abs(c - p - np.exp(-R["r"] * R["T"]) * (eg - K)) < T.GEO_ASIAN_PARITY_ABS


def test_geometric_asian_one_fixing_equals_european():
    c = geometric_asian_price(R["S"], R["K"], R["T"], R["r"], R["sigma"], R["q"], n_fix=1)
    e = float(bs_price(R["S"], R["K"], R["T"], R["r"], R["sigma"], R["q"], "call"))
    assert abs(c - e) < 1e-10


def test_geometric_asian_discrete_converges_to_continuous():
    cont = geometric_asian_price(R["S"], R["K"], R["T"], R["r"], R["sigma"], R["q"], None)
    disc = geometric_asian_price(R["S"], R["K"], R["T"], R["r"], R["sigma"], R["q"], 100_000)
    assert abs(disc - cont) / cont < T.GEO_ASIAN_CONT_LIMIT_REL


def test_geometric_asian_below_european_call_here():
    # averaging lowers variance; for this contract the Asian call must be cheaper
    c = geometric_asian_price(R["S"], R["K"], R["T"], R["r"], R["sigma"], R["q"], 12)
    e = float(bs_price(R["S"], R["K"], R["T"], R["r"], R["sigma"], R["q"], "call"))
    assert 0 < c < e


# ---------- digital ----------
def test_digital_is_limit_of_call_spread():
    h = 1e-4
    cs = (float(bs_price(R["S"], R["K"] - h, R["T"], R["r"], R["sigma"], R["q"], "call"))
          - float(bs_price(R["S"], R["K"] + h, R["T"], R["r"], R["sigma"], R["q"], "call"))) / (2 * h)
    d = digital_call_price(R["S"], R["K"], R["T"], R["r"], R["sigma"], R["q"])
    assert abs(cs - d) < 1e-7
