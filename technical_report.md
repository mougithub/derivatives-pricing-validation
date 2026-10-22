# Independent Validation of Option Pricers and Sensitivity-Based P&L Approximations

**Scope:** European, down-and-out barrier, geometric Asian and digital options under Black-Scholes-Merton (GBM, constant volatility, continuous dividend yield).
**Method:** three independently implemented pricers (analytic, Monte Carlo, Crank-Nicolson PDE) tested against each other and against closed-form benchmarks, with tolerances written before the code was run.
**Status:** complete for the planned scope; items cut for time are listed in Section 7.

---

## 1. Purpose and scope

Questions asked:

1. Do the analytic, Monte Carlo (MC) and PDE pricers agree, and where do they break?
2. Which Greek estimators fail for which payoffs, and how are the failures detected?
3. How far can a delta-gamma(-vega) approximation be trusted against full repricing for a small option book?

Out of scope: stochastic volatility, market calibration, production performance. **Agreement between pricers establishes internal consistency, not correctness against market prices.**

## 2. Models and instruments reviewed

| Instrument | Pricers | Benchmark |
|---|---|---|
| European call / put | analytic, MC (plain, antithetic), PDE | Black-Scholes-Merton |
| Down-and-out call (continuous barrier, no rebate) | closed form, MC (discrete monitoring, with and without BGK shift), PDE | closed form (Reiner-Rubinstein form) |
| Geometric Asian call / put (discrete fixings) | closed form, MC | exact discrete closed form; continuous (Kemna-Vorst) limit |
| Digital (cash-or-nothing) call | closed form, PDE, MC Greeks | closed form |
| Book: short 90-put, short 110-call, long down-and-out call (K=100, H=85) | closed forms (full repricing), PDE cross-check | -- |

Market data: **simulated** (S0=100, r=5%, q=2%, sigma=20%).

## 3. Independent benchmarks

* Vanilla: Black-Scholes closed form; Greeks checked against central finite differences.
* Barrier: the closed form was checked three ways: internal consistency (branch continuity at H=K, H->0 limit, H->S limit, in+out=vanilla), an **independent MC cross-check** (fine monitoring plus BGK shift; z = -0.1 and -1.1 on the two branches), and the continuous-barrier **PDE** (relative difference 3e-6).
* Geometric Asian: exact discrete formula (n=1 equals European; discrete -> continuous limit; put-call parity); MC agrees within 3 SE.
* **Not done:** comparison of any closed form against a published reference table (e.g. Haug). Three independent methods agree with each other, but a shared textbook error is still possible in principle.

## 4. Test plan with pre-set tolerances

All thresholds are in `tests/tolerances.py`, written before the corresponding code existed (Day N block per day). Three kinds of entry are marked in that file and in the README change log: pre-set and unchanged; pre-set but refuted (kept for the record); and **post-hoc** (written after seeing exploratory output). Main criteria: put-call parity < 1e-10; MC within 3 SE at N=1e6; MC CI coverage 93-97%; MC log-log slope -0.55..-0.45; PDE vs analytic < 1e-4 relative; observed PDE order 1.85-2.15 on a smooth payoff; Greeks within 4 SE + 1e-3 relative; delta-gamma flagged "unreliable" when error exceeds 5% of max(|full P&L|, 1% of gross book value).

## 5. Results

**Analytic and MC .** Parity holds to 1e-10 over a 180-point grid. MC matches analytic prices within 3 SE for plain and antithetic sampling. CI coverage over 1000 seeds: 94.5% (plain), 95.3% (antithetic). RMSE slope -0.497 (plain), -0.488 (antithetic) (`figures/mc_convergence.png`). Antithetic SE is 0.75x the plain SE.

**PDE .** Worst relative price error at M=N=800: 3.6e-5 (call), 6.9e-5 (put). Observed order 2.000 (space) and 2.003 (time) on a smooth payoff; 2.001 on the kinked ATM call. Grid delta and gamma within 1.6e-5 and 2.8e-5 (`figures/pde_convergence.png`).

**Known-bad cases .**

* *Barrier monitoring bias:* naive MC overprices by +0.13 to +0.62 on a closed-form price of 7.59 (K=100, H=90) and +0.66 to +2.23 on 6.90 (K=90, H=95) for 400 down to 25 monitoring dates, up to 74 SE; bias decays like n^-0.5. BGK shift removes it (`figures/barrier_monitoring_bias.png`, `barrier_near_barrier.png`).
* *CN on a discontinuous payoff:* with M=800, plain CN is clean at N=800 but at N=100 the digital **gamma error is 20x the peak true gamma** (delta 3.5%). Rannacher smoothing removes it (`figures/digital_pde_oscillation.png`).
* *Pathwise delta on a digital:* returns exactly 0 with SE 0. A smoothed-payoff estimator recovers it within about 1e-4 (`figures/digital_delta.png`).
* *Greeks four ways* on the vanilla call: bump (common random numbers), pathwise, and PDE grid all meet their criteria; a 1% spot bump underestimates ATM gamma by 4.1% at T=0.005 (`figures/greeks_four_ways.png`).

**Delta-gamma vs full repricing .** Book Greeks at the base point (barrier leg by central differences of the closed form, verified against the PDE grid to 5e-6 / 3e-6 / 5e-5 in delta / gamma / vega):

| T (years) | book delta | book gamma | book vega |
|---|---|---|---|
| 1 | +0.456 | -0.0213 | -40.4 |
| 0.25 | +0.464 | -0.0088 | -4.4 |
| 0.05 | +0.511 | **+0.0738** | +7.4 |

## Portfolio-level interpretation

The portfolio analysis illustrates an important distinction between **local sensitivity models** and **full repricing**. For small market moves, the delta-gamma-vega approximation provides a computationally inexpensive estimate of portfolio P&L. In the tested book, approximation error grows approximately cubically with the spot shock, consistent with the truncation of higher-order terms in the sensitivity expansion.

The useful range is therefore **portfolio- and maturity-dependent rather than a universal shock limit**. For the tested book, the spot-shock reliability frontier is roughly **11–12% for longer maturities**, but contracts to approximately **4–5% at \(T=0.05\)**.

From a risk-management perspective, this means a sensitivity-based approximation can be useful for relatively small shocks and fast scenario analysis, while **full repricing becomes increasingly important for larger shocks, short-dated options, and portfolios with strongly nonlinear or path-dependent positions**.

The leg-level results also show why portfolio-level validation matters. Errors in individual positions can partially offset, making the aggregate P&L approximation appear more reliable than some of its components. Conversely, changes in portfolio composition and maturity can materially change the portfolio's gamma and therefore the range over which the approximation remains reliable.

The objective is therefore not to label delta-gamma-vega as simply "accurate" or "inaccurate," but to **identify the operating region in which it meets a specified error criterion and determine when the additional cost of full repricing is justified**.



* **Error scales as the cube of the shock** where the approximation works: fitted slope 3.03 (strangle only) and 3.04 (book), shocks 1-5% (`figures/dg_error_vs_shock.png`).
* **Spot-shock reliability frontier** (smallest |s| with error > 5% of P&L, vol fixed): T=1 and T=0.25: 12.5% down / 11.5% up; T=0.05: 5.0% / 4.5%. With a fixed-denominator metric (error > 2% of gross book value) the frontiers are 12.0% / 13.5% (T=1), 10.0% / 9.5% (T=0.25), 4.5% / 3.0% (T=0.05): the spot conclusion does not depend on the metric.
* **Vol shocks (vega term included, s=0)** are much less reliable. By the pre-set metric the approximation is flagged at about 1.75-2.0 vol points (T=1) and 0.75 vol points (T=0.25); by the 2%-of-gross metric the thresholds are 5-6 (T=1) and 4-5 (T=0.25) vol points. The two metrics disagree because vol-only P&L is small (volga offsets across legs), so the relative metric is unstable. Over the joint box |s| <= 3%, |vol shock| <= 2.5 points, the largest error is 0.8% of gross value (T=1), 1.1% (T=0.25) and **7.4% (T=0.05)**.
* **Share of the full +/-30% x +/-10 vol-point grid within 2% of gross value:** 28% (T=1), 14% (T=0.25), 4% (T=0.05) (`figures/dg_error_heatmap.png`).
* **At T=0.05** the book's gamma turns positive (the barrier call is near expiry and the strangle is nearly worthless), and a quadratic extrapolation then overshoots badly: +25.0 predicted against +8.1 actual for a +20% shock (`figures/dg_pnl_curves.png`).
* **The barrier is not the dominant failure at long maturity.** At T=1 and a -20% shock (below the barrier), the barrier leg's approximation implies an option value of **-1.69** (a long call cannot be negative), but the book-level error is only 1.19 on a P&L of -12.2 (about 10%) because it is partly offset by the short-put leg (+0.85). At T=0.25 the leg errors (+2.9, -2.7, +1.2) net to +1.5 (`figures/dg_leg_errors.png`).

## 6. Findings

Severity reflects the consequence for a user relying on the component without the stated remediation.

| # | Issue | Evidence | Severity | Remediation / use limit |
|---|---|---|---|---|
| F1 | Discretely monitored MC overprices the knock-out call (+1.7% to +32% of price at 400 to 25 dates; up to 74 SE) | `test_barrier_bias.py`; `barrier_monitoring_bias.png` | **High** | Apply the BGK barrier shift, or use more dates; never report naive discrete-monitoring MC as a continuous-barrier price |
| F2 | BGK correction fails within about one shift of the barrier (about 1.6% at 52 dates): error +0.24 at S=91.5; forced price 0 at S=90.5 | `barrier_near_barrier.png` | Medium | Do not use BGK-corrected MC for spots within ~2 shift-widths of the barrier; use the PDE or Brownian-bridge correction |
| F3 | Plain CN produces gamma oscillations on a digital payoff at coarse time grids (gamma error 20x peak at N=100) while looking clean at N=800 | `test_digital_pde.py`; `digital_pde_oscillation.png` | **High** (for hedging Greeks) | Rannacher start-up steps on every discontinuous payoff; test Greeks, not only prices, and test more than one time-step count |
| F4 | Valuing the strike node as 0 or 1 instead of the cell average 0.5 leaves ~0.3% delta / 1.5% gamma error that Rannacher does not remove | exploration table in README change log | Medium | Cell-average the initial condition at a discontinuity |
| F5 | Pathwise delta on a digital returns exactly 0 with SE 0 | `test_greeks.py::test_digital_pathwise_delta_fails_silently` | **High** (silent) | Smoothed payoff (used, 1% ramp) or likelihood-ratio estimator (not built); report the smoothing bias |
| F6 | Fixed-percentage spot bump underestimates ATM gamma by 4.1% at T=0.005 | `test_greeks.py` near-expiry test | Medium | Scale the bump with S*sigma*sqrt(T) |
| F7 | Delta-gamma is unreliable beyond roughly +/-11-12% spot moves for T >= 0.25 and beyond +/-4-5% at T=0.05; error grows with the cube of the shock | `dg_error_vs_shock.png`, `dg_pnl_curves.png` | **High** for large-move stress use | Use full repricing for stress shocks above ~10% and for any book with short-dated options |
| F8 | Reliability under vol shocks depends on the metric used (about 1-2 vs 5-6 vol points at T=1) because vol-only P&L is small | Section 5; `dg_error_heatmap.png` | Medium | Report both a P&L-relative and a gross-value-relative error; do not rely on one |
| F9 | Leg-level approximation errors offset in the book; the barrier leg can imply a negative option value below the barrier | `dg_leg_errors.png` | Medium | Check the approximation per position or per risk bucket, not only on the net book |
| F10 | Smallest margin to a pre-set tolerance: PDE put price error 6.9e-5 against 1e-4 | `test_pde.py` | Low | Coarser grids will fail first for out-of-the-money puts |
| F11 | Unexplained: BGK-shifted MC is 2.4 SE below the closed form at 400 dates (K=100, H=90). Among 10 BGK comparisons the chance that at least one exceeds 2.4 SE by luck is about 15%, so this is consistent with chance but not proven | README Day 3 table | Low | Re-run with more paths if it matters |

Validation-process notes (not model findings): a pre-set expectation was refuted (plain CN was expected to fail at M=N=800); the coverage test as originally specified (200 seeds, 93-97%) would fail a correct pricer about 14% of the time; and four of my own test-code bugs (continuity-test step size, BGK shift sign, time-order spatial floor, digital-gamma finite-difference step) were caught by the validation itself and fixed with tolerances unchanged. All are in the README change log.

## 7. Limitations

* Simulated inputs and GBM with constant volatility; no calibration to market prices. Results about approximation error depend on the chosen book, strikes and shock grid.
* Shocks are instantaneous (no time decay); the approximation uses delta, gamma and vega only, with no vanna or volga terms, which partly explains the vol-shock errors.
* The book's barrier Greeks use finite differences of a closed form, not an analytic formula.
* **Not done (scope cut):** arithmetic Asian (MC with geometric control variate), likelihood-ratio digital estimator, Greeks near the barrier compared across estimators, PDE-grid Greeks for the barrier option beyond the base point, and comparison to a published reference table.
* Three post-hoc choices: the coarse-N digital thresholds, the 2%-of-gross contour in the heatmap, and the alternate metric used for sensitivity checks. They were chosen after seeing exploratory output.

## 8. Open questions

The delta-gamma approximation worked well for moves up to about 10% when options were not short-dated. Its error grew with the cube of the shock, it deteriorated sharply at T=0.05, where the book's gamma changed sign, and its P&L-relative error metric was unstable whenever spot and vol effects offset. The barrier itself was not the main cause of failure at long maturity, because errors in different legs partly cancelled.

Fixed-income risk is also managed with a second-order sensitivity approximation (duration and convexity), often applied to curve-shaped shocks rather than a single factor. Does the same pattern appear there: error growing with the cube of the shock, much earlier failure when the instruments are short-dated, and offsetting errors across positions that make the portfolio-level number look better than the position-level ones? And does a non-parallel shock (slope, curvature) break the approximation at a smaller size than a parallel one?

## 9. Conclusion and recommended use limits

The three pricers agree with each other and with the closed forms within the pre-set tolerances for vanilla, geometric Asian and (continuously monitored) barrier options, with the known-bad cases below reproduced and fixed.

Recommended use limits: (1) do not use discretely monitored MC without a barrier correction, and not at all within a few shift-widths of the barrier; (2) use Rannacher-smoothed CN, with a cell-averaged payoff, for any discontinuous payoff, and test Greeks at more than one time-step count; (3) do not use pathwise delta on discontinuous payoffs; (4) treat delta-gamma-vega as reliable only for spot moves within about 10% and vol moves within a couple of points, and for T=0.05 only within about 3-4%; beyond that, use full repricing.
