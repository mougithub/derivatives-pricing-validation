import numpy as np
import pytest
import tolerances as T
from black_scholes import bs_price, bs_greeks
from pde_pricer import pde_price
from convergence import pde_order_study, loglog_slope

R = T.REF
BASE = dict(K=R["K"], T=R["T"], r=R["r"], sigma=R["sigma"], q=R["q"])


def _pde(kind, S0, **kw):
    return pde_price(kind, S0, **BASE, M=kw.pop("M", T.PDE_FINE_M), N=kw.pop("N", T.PDE_FINE_N),
                     n_sd=kw.pop("n_sd", T.PDE_N_SD), **kw)


@pytest.mark.parametrize("rannacher", [False, True])
@pytest.mark.parametrize("kind", ["call", "put"])
def test_pde_price_vs_analytic(kind, rannacher):
    worst = 0.0
    for S0 in T.PDE_SPOTS:
        ex = float(bs_price(S0, R["K"], R["T"], R["r"], R["sigma"], R["q"], kind))
        worst = max(worst, abs(_pde(kind, S0, rannacher=rannacher)["price"] - ex) / ex)
    print(f"max rel price error {kind} rannacher={rannacher}: {worst:.2e}")
    assert worst < T.PDE_PRICE_REL, worst


def test_pde_put_call_parity():
    for S0 in T.PDE_SPOTS:
        c, p = _pde("call", S0)["price"], _pde("put", S0)["price"]
        ref = S0 * np.exp(-R["q"] * R["T"]) - R["K"] * np.exp(-R["r"] * R["T"])
        assert abs(c - p - ref) < T.PDE_PARITY_ABS, (S0, c - p - ref)


@pytest.mark.parametrize("rannacher", [False, True])
@pytest.mark.parametrize("kind", ["call", "put"])
def test_pde_delta_gamma_vs_analytic(kind, rannacher):
    wd = wg = 0.0
    for S0 in T.PDE_SPOTS:
        g = bs_greeks(S0, R["K"], R["T"], R["r"], R["sigma"], R["q"], kind)
        res = _pde(kind, S0, rannacher=rannacher)
        wd = max(wd, abs(res["delta"] - float(g["delta"])) / abs(float(g["delta"])))
        wg = max(wg, abs(res["gamma"] - float(g["gamma"])) / float(g["gamma"]))
    print(f"{kind} rannacher={rannacher}: max rel delta err {wd:.2e}, gamma err {wg:.2e}")
    assert wd < T.PDE_DELTA_REL, wd
    assert wg < T.PDE_GAMMA_REL, wg


def test_pde_domain_truncation_is_negligible():
    # same dx, wider domain: M scales with n_sd
    for kind in ("call", "put"):
        a = _pde(kind, 100.0, n_sd=6.0, M=800)["price"]
        b = _pde(kind, 100.0, n_sd=9.0, M=1200)["price"]
        assert abs(a - b) < T.PDE_DOMAIN_ABS, (kind, a - b)


def test_observed_order_smooth_payoff():
    st = pde_order_study(R["r"], R["sigma"], R["q"], R["T"], R["K"], 2.0,
                         T.PDE_SPACE_M_GRID, T.PDE_SPACE_N, T.PDE_TIME_N_GRID, T.PDE_TIME_M,
                         time_ref_N=T.PDE_TIME_REF_N)
    print(f"observed order: space {st['order_space']:.3f}, time {st['order_time']:.3f}")
    assert T.PDE_ORDER_LOW <= st["order_space"] <= T.PDE_ORDER_HIGH, st["order_space"]
    assert T.PDE_ORDER_LOW <= st["order_time"] <= T.PDE_ORDER_HIGH, st["order_time"]


def test_observed_order_kinked_vanilla_reported():
    ex = float(bs_price(100.0, R["K"], R["T"], R["r"], R["sigma"], R["q"], "call"))
    ns = [100, 200, 400, 800]
    errs = [abs(_pde("call", 100.0, M=n, N=n)["price"] - ex) / ex for n in ns]
    order = -loglog_slope(ns, errs)
    print(f"ATM vanilla call, M=N refined: errors {['%.1e' % e for e in errs]}, order {order:.3f}")
    assert order >= T.PDE_KINK_ORDER_MIN, order
