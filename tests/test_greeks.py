"""Greeks four ways on the vanilla call, plus the digital pathwise failure and fix."""
import numpy as np
import pytest
import tolerances as T
from black_scholes import bs_greeks
from exotics import digital_call_greeks
import greeks as G

R = T.REF
A = dict(K=R["K"], T=R["T"], r=R["r"], sigma=R["sigma"], q=R["q"])


def _ref(S, **over):
    a = dict(A); a.update(over)
    return bs_greeks(S, a["K"], a["T"], a["r"], a["sigma"], a["q"], "call")


@pytest.mark.parametrize("S", T.GREEKS_SPOTS)
def test_bump_crn_vs_analytic(S):
    est = G.mc_call_greeks_bump(S, **A, n_paths=T.GREEKS_N_PATHS, h_rel=T.GREEKS_BUMP_REL,
                                vol_bump=T.GREEKS_VOL_BUMP, seed=[T.SEED, 10, int(S)])
    ref = _ref(S)
    for name in ("delta", "gamma", "vega"):
        e = est[name]
        assert e.within(float(ref[name]), T.GREEKS_Z_MAX, T.GREEKS_BIAS_REL), (S, name, e, float(ref[name]))


@pytest.mark.parametrize("S", T.GREEKS_SPOTS)
def test_pathwise_vs_analytic(S):
    est = G.mc_call_greeks_pathwise(S, **A, n_paths=T.GREEKS_N_PATHS, seed=[T.SEED, 11, int(S)])
    ref = _ref(S)
    for name in ("delta", "vega"):
        assert est[name].within(float(ref[name]), T.GREEKS_Z_MAX, 0.0), (S, name, est[name], float(ref[name]))


def test_pde_grid_greeks_incl_vega_vs_analytic():
    wv = 0.0
    for S in T.GREEKS_SPOTS:
        g = G.pde_vanilla_greeks(S, **A, M=T.PDE_FINE_M, N=T.PDE_FINE_N)
        wv = max(wv, abs(g["vega"] - float(_ref(S)["vega"])) / float(_ref(S)["vega"]))
    print(f"PDE vega max rel error: {wv:.2e}")
    assert wv < T.PDE_VEGA_REL, wv


# ---------- near expiry ----------
def test_near_expiry_fixed_bump_gamma_is_biased_adaptive_bump_is_not():
    S, Tn = 100.0, T.NEAR_EXPIRY_T
    a = dict(A); a["T"] = Tn
    ref = float(_ref(S, T=Tn)["gamma"])
    fixed = G.mc_call_greeks_bump(S, **a, n_paths=T.GREEKS_N_PATHS, h_rel=T.GREEKS_BUMP_REL, seed=[T.SEED, 12, 0])["gamma"]
    h_ad = T.NEAR_EXPIRY_ADAPTIVE_H * S * R["sigma"] * np.sqrt(Tn)
    adapt = G.mc_call_greeks_bump(S, **a, n_paths=T.GREEKS_N_PATHS, h=h_ad, seed=[T.SEED, 12, 1])["gamma"]
    print(f"near-expiry gamma ref {ref:.4f}: fixed 1% bump {fixed.value:.4f} (SE {fixed.se:.4f}, "
          f"{(fixed.value-ref)/ref:+.1%}); adaptive h={h_ad:.3f} {adapt.value:.4f} (SE {adapt.se:.4f}, {(adapt.value-ref)/ref:+.1%})")
    assert not fixed.within(ref, T.GREEKS_Z_MAX, T.GREEKS_BIAS_REL)      # known-bad reproduced
    assert adapt.within(ref, T.GREEKS_Z_MAX, T.GREEKS_BIAS_REL)           # fix works


def test_near_expiry_pde_gamma():
    S, Tn = 100.0, T.NEAR_EXPIRY_T
    g = G.pde_vanilla_greeks(S, K=R["K"], T=Tn, r=R["r"], sigma=R["sigma"], q=R["q"])
    ref = float(_ref(S, T=Tn)["gamma"])
    print(f"near-expiry PDE gamma rel error: {abs(g['gamma']-ref)/ref:.2e}")
    assert abs(g["gamma"] - ref) / ref < T.NEAR_EXPIRY_PDE_GAMMA_REL


# ---------- digital delta ----------
@pytest.mark.parametrize("S", T.DIGITAL_GREEKS_SPOTS)
def test_digital_pathwise_delta_fails_silently(S):
    ref = float(digital_call_greeks(S, **A)["delta"])
    est = G.mc_digital_delta_pathwise_naive(S, **A, n_paths=T.GREEKS_N_PATHS, seed=[T.SEED, 13, int(S)])
    assert abs(est.value) < T.DIGITAL_PATHWISE_ZERO_ABS and est.se == 0.0 and ref > 1e-3


@pytest.mark.parametrize("S", T.DIGITAL_GREEKS_SPOTS)
def test_digital_smoothed_pathwise_and_bump_delta(S):
    ref = float(digital_call_greeks(S, **A)["delta"])
    sm = G.mc_digital_delta_pathwise_smoothed(S, **A, n_paths=T.GREEKS_N_PATHS, eps_rel=T.DIGITAL_SMOOTH_EPS_REL,
                                              seed=[T.SEED, 14, int(S)])
    bu = G.mc_digital_delta_bump(S, **A, n_paths=T.GREEKS_N_PATHS, h_rel=T.GREEKS_BUMP_REL, seed=[T.SEED, 15, int(S)])
    print(f"S={S}: digital delta ref {ref:.5f} | smoothed {sm.value:.5f} (SE {sm.se:.5f}) | bump {bu.value:.5f} (SE {bu.se:.5f})")
    assert sm.within(ref, T.GREEKS_Z_MAX, T.DIGITAL_BIAS_REL), (sm, ref)
    assert bu.within(ref, T.GREEKS_Z_MAX, T.DIGITAL_BIAS_REL), (bu, ref)
