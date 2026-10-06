"""Known-bad case #2: Crank-Nicolson oscillations on a discontinuous payoff, and Rannacher smoothing."""
import numpy as np
import pytest
import tolerances as T
from exotics import digital_call_price, digital_call_greeks
from pde_pricer import pde_price, grid_greeks

R = T.REF
ARGS = (R["K"], R["T"], R["r"], R["sigma"], R["q"])


def _node_errors(rannacher, N):
    res = pde_price("digital", R["K"], *ARGS, M=T.PDE_FINE_M, N=N, rannacher=rannacher)
    xi, delta, gamma = grid_greeks(res["x"], res["V"])
    S = np.exp(xi)
    w = np.abs(xi - np.log(R["K"])) <= T.DIGITAL_WINDOW
    ex = digital_call_greeks(S[w], *ARGS)
    nd = np.max(np.abs(delta[w] - ex["delta"])) / np.max(np.abs(ex["delta"]))
    ng = np.max(np.abs(gamma[w] - ex["gamma"])) / np.max(np.abs(ex["gamma"]))
    return nd, ng


def test_analytic_digital_greeks_vs_finite_difference():
    f = lambda s: digital_call_price(s, *ARGS)
    for S in (90.0, 100.0, 110.0):
        g = digital_call_greeks(S, *ARGS)
        h = 0.01
        d_fd = (f(S + h) - f(S - h)) / (2 * h)
        h2 = 0.02   # (0.1 was too coarse: FD truncation error ~1e-5 for a digital payoff)
        g_fd = (f(S + h2) - 2 * f(S) + f(S - h2)) / h2**2
        assert abs(float(g["delta"]) - d_fd) / abs(d_fd) < T.GREEKS_FD_REL
        assert abs(float(g["gamma"]) - g_fd) / abs(g_fd) < T.GREEKS_FD_REL


def test_fine_time_grid_plain_cn_is_already_clean():
    # Characterisation (post-hoc): the pre-set expectation that plain CN misbehaves at M=N=800 was REFUTED.
    nd, ng = _node_errors(False, T.PDE_FINE_N)
    print(f"plain CN, digital, M=N=800: normalised max error delta {nd:.1e}, gamma {ng:.1e}")
    assert nd <= T.DIGITAL_FINE_CN_MAX and ng <= T.DIGITAL_FINE_CN_MAX, (nd, ng)


def test_plain_cn_gamma_oscillates_on_coarse_time_grid():
    nd, ng = _node_errors(False, T.DIGITAL_COARSE_N)
    print(f"plain CN, digital, M=800 N={T.DIGITAL_COARSE_N}: normalised max error delta {nd:.3f}, gamma {ng:.1f}")
    assert ng >= T.DIGITAL_CN_GAMMA_MIN_COARSE, ng


@pytest.mark.parametrize("N", [T.DIGITAL_COARSE_N, T.PDE_FINE_N])
def test_rannacher_removes_oscillation(N):
    nd, ng = _node_errors(True, N)
    print(f"CN + Rannacher, digital, N={N}: normalised max error delta {nd:.1e}, gamma {ng:.1e}")
    assert nd <= T.DIGITAL_RAN_DELTA_MAX and ng <= T.DIGITAL_RAN_GAMMA_MAX, (nd, ng)


def test_rannacher_digital_price():
    for S0 in (95.0, 100.0, 105.0):
        ex = digital_call_price(S0, *ARGS)
        p = pde_price("digital", S0, *ARGS, M=T.PDE_FINE_M, N=T.PDE_FINE_N, rannacher=True)["price"]
        assert abs(p - ex) / ex < T.DIGITAL_PRICE_REL, (S0, p, ex)
