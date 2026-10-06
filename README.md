# Derivatives pricing validation

Independent validation of option pricers (analytic, Monte Carlo, PDE) against closed-form benchmarks.
**Status: Day 1 of 4 (analytic + Monte Carlo layers).** PDE, known-bad cases, Greeks and the
delta-gamma test follow.

**Out of scope:** stochastic volatility, market calibration, production-grade performance.
Agreement between pricers shows internal consistency, not correctness against the market.

## Run
```
pip install -r requirements.txt
pytest -s            # ~15 s; -s prints coverage / slope / SE-ratio values
```
All tolerances live in `tests/tolerances.py` and were written before any test was run.
Seeds are derived from `SEED = 20261005` plus a test id.

## Day 1 results
| Check | Criterion (pre-set) | Observed |
|---|---|---|
| Put-call parity, 180-point grid | abs < 1e-10 | pass |
| Price bounds, monotonicity, sigma/T limits | see `tolerances.py` | pass |
| Analytic delta/gamma/vega vs central FD | rel < 1e-5 | pass |
| Barrier: branch continuity at H=K, H->0, H->S, in+out=vanilla | 1e-9 / 1e-8 / 1e-3 | pass |
| Barrier closed form vs independent MC (BGK, 200 steps), both branches | within 4 SE | z = -0.1, -1.1 |
| Geometric Asian: parity, n=1 equals European, discrete -> continuous | 1e-10 / 1e-10 / 1e-3 rel | pass |
| MC vs analytic (N=1e6, plain and antithetic, call and put) | within 3 SE | pass |
| MC 95% CI coverage (1000 seeds x 20k paths) | 93%-97% | plain 94.5%, antithetic 95.3% |
| MC RMSE log-log slope | -0.55 to -0.45 | plain -0.497, antithetic -0.488 |
| Antithetic SE / plain SE (equal N) | < 0.95 | 0.747 |
| Geometric Asian MC vs exact discrete closed form | within 3 SE | pass |

Figure: `figures/mc_convergence.png`.

## Notes on the pricers
* Antithetic SE is computed on pair averages (the i.i.d. units), not on all 2m payoffs.
* Geometric Asian is exact for discrete monitoring (fixings at iT/n); `n_fix=None` gives the Kemna-Vorst continuous limit.
* Barrier closed form is continuous monitoring, zero rebate. **TODO before relying on it:** check one
  value against a trusted reference (e.g. Haug's tables). Day 1 evidence is internal consistency plus
  the independent MC cross-check above.

## Change log
* **Day 1 (before running):** coverage test uses 1000 seeds instead of 200. With 200 seeds the binomial
  SD of observed coverage is 1.5 points, so a correct pricer fails a 93-97% band about 14% of the time;
  with 1000 seeds it is about 0.3%.
* **Day 1 (test fix, tolerances unchanged):** barrier continuity test compared points 2e-9 apart on a
  curve with slope -0.7, so the test itself produced a 1.35e-9 gap; points are now 2e-12 apart.
* **Day 1 (test fix, tolerances unchanged):** BGK barrier shift in the MC cross-check had the wrong sign.
  To approximate a continuous down barrier with discrete monitoring, shift the barrier toward spot
  (`H*exp(+0.5826*sigma*sqrt(dt))`). The wrong sign doubled the bias and was caught by the cross-check.
