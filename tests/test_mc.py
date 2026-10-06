import numpy as np
import pytest
import tolerances as T
from black_scholes import bs_price
from exotics import geometric_asian_price
from mc_pricer import mc_european, mc_asian
from convergence import mc_rmse_vs_n, loglog_slope

R = T.REF
EXACT = float(bs_price(**R, kind="call"))


@pytest.mark.parametrize("antithetic", [False, True])
@pytest.mark.parametrize("kind", ["call", "put"])
def test_mc_vs_analytic_within_3_se(antithetic, kind):
    res = mc_european(**R, kind=kind, n_paths=T.MC_N_PATHS, antithetic=antithetic,
                      seed=[T.SEED, 1, int(antithetic), kind == "put"])
    exact = float(bs_price(**R, kind=kind))
    assert abs(res.price - exact) / res.se < T.MC_Z_MAX, (res, exact)


@pytest.mark.parametrize("antithetic", [False, True])
def test_ci_coverage(antithetic):
    hits = sum(
        mc_european(**R, kind="call", n_paths=T.COVERAGE_N_PATHS, antithetic=antithetic,
                    seed=[T.SEED, 4, int(antithetic), i]).contains(EXACT)
        for i in range(T.COVERAGE_N_SEEDS))
    cov = hits / T.COVERAGE_N_SEEDS
    print(f"coverage antithetic={antithetic}: {cov:.3f}")
    assert T.COVERAGE_LOW <= cov <= T.COVERAGE_HIGH, cov


@pytest.mark.parametrize("antithetic", [False, True])
def test_convergence_slope(antithetic):
    rmse = mc_rmse_vs_n(R, T.SLOPE_N_GRID, T.SLOPE_REPS, antithetic, T.SEED)
    slope = loglog_slope(T.SLOPE_N_GRID, rmse)
    print(f"slope antithetic={antithetic}: {slope:.3f}")
    assert T.SLOPE_LOW <= slope <= T.SLOPE_HIGH, slope


def test_antithetic_reduces_standard_error():
    n = 200_000
    plain = mc_european(**R, n_paths=n, antithetic=False, seed=[T.SEED, 5, 0])
    anti = mc_european(**R, n_paths=n, antithetic=True, seed=[T.SEED, 5, 1])
    ratio = anti.se / plain.se
    print(f"SE ratio antithetic/plain: {ratio:.3f}")
    assert ratio < T.ANTITHETIC_SE_RATIO_MAX


def test_antithetic_requires_even_paths():
    with pytest.raises(ValueError):
        mc_european(**R, n_paths=1001, antithetic=True, seed=0)


@pytest.mark.parametrize("antithetic", [False, True])
def test_geometric_asian_mc_vs_closed_form(antithetic):
    exact = geometric_asian_price(R["S"], R["K"], R["T"], R["r"], R["sigma"], R["q"],
                                  n_fix=T.MC_ASIAN_N_FIX)
    res = mc_asian(R["S"], R["K"], R["T"], R["r"], R["sigma"], R["q"], n_fix=T.MC_ASIAN_N_FIX,
                   average="geometric", n_paths=T.MC_N_PATHS, antithetic=antithetic,
                   seed=[T.SEED, 6, int(antithetic)])
    assert abs(res.price - exact) / res.se < T.MC_Z_MAX, (res, exact)
