"""Day 4: how far can a delta-gamma(-vega) approximation be trusted against full repricing?"""
from functools import lru_cache
import numpy as np
import pytest
import tolerances as T
from convergence import loglog_slope
from pde_pricer import pde_down_and_out_call
import greeks as G
from greeks import Position

R = T.REF
M = dict(S=R["S"], r=R["r"], sigma=R["sigma"], q=R["q"])
PUT = Position("put", T.BOOK_STRANGLE_K[0], -1.0)
CALL = Position("call", T.BOOK_STRANGLE_K[1], -1.0)
DOC = Position("doc", T.BOOK_DOC_K, +1.0, T.BOOK_DOC_H)
STRANGLE = (PUT, CALL)
BOOK = (PUT, CALL, DOC)


def _metric(book, Tm, s, v=0.0, use_vega=True):
    g = G.book_greeks(book, M["S"], Tm, M["r"], M["sigma"], M["q"])
    gross = G.gross_value(book, M["S"], Tm, M["r"], M["sigma"], M["q"])
    dS = s * M["S"]
    a = G.approx_pnl(g, dS, v, use_vega)
    f = G.full_pnl(book, M["S"], Tm, M["r"], M["sigma"], M["q"], dS, v)
    return G.flag_metric(a, f, gross, T.DG_FLOOR_FRAC), a, f


def test_zero_shock_identity():
    for Tm in T.BOOK_T_GRID:
        m, a, f = _metric(BOOK, Tm, 0.0, 0.0)
        assert a == 0.0 and abs(f) < 1e-12


def test_doc_greeks_closed_form_fd_vs_pde_grid():
    Tm = 1.0
    S, K, H = M["S"], T.BOOK_DOC_K, T.BOOK_DOC_H
    g = G.book_greeks((DOC,), S, Tm, M["r"], M["sigma"], M["q"])
    base = pde_down_and_out_call(S, K, H, Tm, M["r"], M["sigma"], M["q"])
    xm = base["x_max"]   # same grid for the vol bump
    up = pde_down_and_out_call(S, K, H, Tm, M["r"], M["sigma"] + 0.005, M["q"], x_max=xm)["price"]
    dn = pde_down_and_out_call(S, K, H, Tm, M["r"], M["sigma"] - 0.005, M["q"], x_max=xm)["price"]
    vega = (up - dn) / 0.01
    rd, rg, rv = (abs(base["delta"] - g["delta"]) / abs(g["delta"]), abs(base["gamma"] - g["gamma"]) / abs(g["gamma"]),
                  abs(vega - g["vega"]) / abs(g["vega"]))
    print(f"DOC Greeks closed-form FD vs PDE grid: delta {rd:.1e}, gamma {rg:.1e}, vega {rv:.1e}")
    assert rd < T.BOOK_GREEKS_PDE_DELTA_REL and rg < T.BOOK_GREEKS_PDE_GAMMA_REL and rv < T.BOOK_GREEKS_PDE_VEGA_REL


def test_full_repricing_engine_matches_pde_over_shock_range():
    K, H = T.BOOK_DOC_K, T.BOOK_DOC_H
    worst = 0.0
    for v in (-0.10, 0.0, 0.10):
        for s in (-0.05, 0.0, 0.05, 0.10, 0.20, 0.30):
            S = M["S"] * (1 + s)
            if S < 1.05 * H:
                continue
            cf = G.position_value(DOC, S, 1.0, M["r"], M["sigma"] + v, M["q"])
            pd = pde_down_and_out_call(S, K, H, 1.0, M["r"], M["sigma"] + v, M["q"])["price"]
            worst = max(worst, abs(pd - cf) / cf)
    print(f"full-repricing engine vs PDE, worst rel diff over shock range: {worst:.1e}")
    assert worst < T.FULL_REPRICE_PDE_REL


def _slope(book, Tm):
    errs = []
    for s in T.DG_SLOPE_SHOCKS:
        e = []
        for sign in (-1, 1):
            _, a, f = _metric(book, Tm, sign * s, 0.0, use_vega=False)
            e.append(abs(a - f))
        errs.append(max(e))
    return -(-loglog_slope(T.DG_SLOPE_SHOCKS, errs)), errs


def test_error_slope_strangle_is_cubic():
    sl, errs = _slope(STRANGLE, 1.0)
    print(f"strangle-only delta-gamma error slope vs |s| (T=1): {sl:.3f}")
    assert T.DG_SLOPE_STRANGLE[0] <= sl <= T.DG_SLOPE_STRANGLE[1], sl


def test_error_slope_book():
    sl, errs = _slope(BOOK, 1.0)
    print(f"book delta-gamma error slope vs |s| (T=1): {sl:.3f}")
    assert T.DG_SLOPE_BOOK[0] <= sl <= T.DG_SLOPE_BOOK[1], sl


def test_reliable_for_small_moves_t1():
    for s in T.SPOT_SHOCKS:
        if s <= T.DG_RELIABLE_S:
            for sign in (-1, 1):
                m, _, _ = _metric(BOOK, 1.0, sign * s, 0.0, use_vega=False)
                assert m <= T.DG_FLAG_REL, (sign * s, m)


def test_fails_below_barrier_and_on_some_upside_shock_t1():
    m_dn, a, f = _metric(BOOK, 1.0, -0.20, 0.0, use_vega=False)
    print(f"T=1, s=-20% (below barrier): approx {a:+.3f} vs full {f:+.3f}, metric {m_dn:.2f}")
    assert m_dn > T.DG_FLAG_REL
    ups = [_metric(BOOK, 1.0, s, 0.0, use_vega=False)[0] for s in T.SPOT_SHOCKS]
    print("T=1 upside metrics:", [f"{u:.3f}" for u in ups])
    assert max(ups) > T.DG_FLAG_REL


def test_frontier_shrinks_with_maturity():
    fr = {}
    for Tm in T.BOOK_T_GRID:
        fr[Tm] = G.reliability_frontier(BOOK, M["S"], Tm, M["r"], M["sigma"], M["q"], T.DG_FLOOR_FRAC,
                                        T.DG_FLAG_REL, T.DG_FRONTIER_STEP, T.DG_FRONTIER_MAX)
        print(f"T={Tm}: reliability frontier (down, up) = {fr[Tm]}")
    assert min(fr[0.05]) < min(fr[1.0])
