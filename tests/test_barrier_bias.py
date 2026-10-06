"""Known-bad case #1: discrete monitoring bias in MC, its BGK fix, and the continuous-barrier PDE."""
from functools import lru_cache
import numpy as np
import pytest
import tolerances as T
from exotics import down_and_out_call
from mc_pricer import mc_down_and_out_call_pair
from pde_pricer import pde_down_and_out_call
from convergence import loglog_slope

R = T.REF


@lru_cache(maxsize=None)
def _study(K, H):
    cf = down_and_out_call(R["S"], K, H, R["T"], R["r"], R["sigma"], R["q"])
    rows = []
    for n in T.BARRIER_STEPS_GRID:
        naive, bgk = mc_down_and_out_call_pair(R["S"], K, H, R["T"], R["r"], R["sigma"], R["q"], n_steps=n,
                                               n_paths=T.BARRIER_BIAS_N_PATHS,
                                               seed=[T.SEED, 7, int(K), int(H), n])
        rows.append(dict(n=n, naive=naive.price - cf, naive_se=naive.se, bgk=bgk.price - cf, bgk_se=bgk.se))
    print(f"\nK={K} H={H} closed form={cf:.4f}")
    for w in rows:
        print(f"  n={w['n']:4d} naive bias {w['naive']:+.4f} ({w['naive']/w['naive_se']:+.1f} SE) | "
              f"BGK error {w['bgk']:+.4f} ({w['bgk']/w['bgk_se']:+.1f} SE)")
    return cf, rows


@pytest.mark.parametrize("K,H", T.BARRIER_CASES)
def test_naive_discrete_mc_is_biased_high(K, H):
    _, rows = _study(K, H)
    assert all(w["naive"] > 0 for w in rows)
    w50 = [w for w in rows if w["n"] == 50][0]
    assert w50["naive"] / w50["naive_se"] > T.BARRIER_NAIVE_Z_MIN, w50


@pytest.mark.parametrize("K,H", T.BARRIER_CASES)
def test_naive_bias_shrinks_like_root_dt(K, H):
    _, rows = _study(K, H)
    slope = loglog_slope([w["n"] for w in rows], [w["naive"] for w in rows])
    print(f"K={K} H={H} naive bias slope vs n: {slope:.3f}")
    assert T.BARRIER_NAIVE_SLOPE_LOW <= slope <= T.BARRIER_NAIVE_SLOPE_HIGH, slope


@pytest.mark.parametrize("K,H", T.BARRIER_CASES)
def test_bgk_shift_removes_most_of_the_bias(K, H):
    _, rows = _study(K, H)
    for w in rows:
        if w["n"] >= 50:
            assert abs(w["bgk"]) <= T.BARRIER_BGK_FRACTION * w["naive"] + 3 * w["bgk_se"], w


@pytest.mark.parametrize("rannacher", [False, True])
def test_pde_continuous_barrier_vs_closed_form(rannacher):
    K, H = 100.0, 90.0
    worst = 0.0
    for S0 in T.BARRIER_PDE_SPOTS:
        cf = down_and_out_call(S0, K, H, R["T"], R["r"], R["sigma"], R["q"])
        pde = pde_down_and_out_call(S0, K, H, R["T"], R["r"], R["sigma"], R["q"], rannacher=rannacher)["price"]
        worst = max(worst, abs(pde - cf) / cf)
    print(f"PDE barrier max rel error rannacher={rannacher}: {worst:.2e}")
    assert worst < T.BARRIER_PDE_REL, worst
