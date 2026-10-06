# Derivatives Pricing & Model Validation

A quantitative validation framework for European, barrier, Asian, and digital options. The project compares analytic, Monte Carlo, and finite-difference PDE methods, tests numerical failure modes, validates Greek estimators, and evaluates when a delta-gamma-vega approximation remains reliable for portfolio stress testing.

**Focus:** numerical methods, model validation, sensitivity analysis, and risk approximation.

> **Scope:** The project emphasizes numerical validation, convergence behavior, model sensitivities, and stress-testing reliability across a range of pricing and risk scenarios.

---

## Project progression

The framework was developed incrementally, adding independent methods and validation layers:

**Analytic benchmarks**
Closed-form Black–Scholes prices and Greeks provide deterministic reference values and baseline consistency checks.

**Monte Carlo pricing**
Monte Carlo implementations add confidence intervals, convergence analysis, antithetic variance reduction, and validation against analytic benchmarks. Geometric Asian options provide an additional independent test case.

**Finite-difference PDE pricing**
A Crank–Nicolson solver in log-price space provides a third pricing approach, with spatial and temporal convergence studies and numerical delta/gamma estimates.

**Numerical failure-mode testing**
The framework deliberately tests cases where standard numerical methods can fail, including discrete barrier-monitoring bias, discontinuous digital payoffs, and pathwise differentiation of digital options.

**Greek validation**
Bump-and-revalue, pathwise, PDE, and analytic approaches are compared for delta, gamma, and vega, including near-expiry sensitivity to bump size.

**Portfolio stress testing**
The individual pricing and Greek components are combined into a portfolio-level delta-gamma-vega approximation and compared against full repricing across spot shocks, volatility shocks, and maturities.

---

## Key validation results

* Monte Carlo RMSE convergence follows the expected (N^{-1/2}) behavior, with fitted slopes of approximately **-0.50**.
* Antithetic sampling reduces standard error to **0.747×** the plain-Monte-Carlo value at equal path count.
* Crank–Nicolson shows approximately **second-order spatial and temporal convergence** in the tested smooth regime.
* Vanilla PDE prices agree with analytic benchmarks to better than **(7\times10^{-5})** relative error in the tested cases.
* PDE delta/gamma errors remain below approximately **(3\times10^{-5})** for the tested vanilla cases.
* Barrier Monte Carlo demonstrates substantial discrete-monitoring bias, reaching as high as **74 standard errors** in the near-barrier test; the BGK correction removes most of the bias away from the barrier.
* Four Greek-estimation approaches are compared: **analytic, bump-and-revalue, pathwise, and PDE**.
* Near expiry, a fixed 1% spot bump produces a measurable gamma bias, while scaling the bump with (S\sigma\sqrt{T}) substantially improves the estimate.
* For the portfolio stress test, delta-gamma-vega approximation error scales approximately with the **cube of the spot shock**, with fitted slopes of **3.03–3.04**.
* The tested approximation remains reliable over a larger shock range for longer maturities, while the reliability frontier contracts substantially at **(T=0.05)**.

Detailed numerical results, tolerances, tables, and validation methodology are documented in **[`report.md`](report.md)**.

---

## Important failure cases

The project intentionally includes cases where a numerical method gives a misleading result.

### Discrete barrier monitoring

Naive discrete monitoring systematically overprices a continuously monitored down-and-out option because barrier crossings between monitoring dates are missed. The **Broadie–Glasserman–Kou correction** substantially reduces this bias, but becomes unreliable when the spot is very close to the shifted barrier.

### Crank–Nicolson with digital payoffs

Plain Crank–Nicolson can develop severe oscillations for discontinuous digital payoffs when the time grid is too coarse relative to the spatial grid. In the tested coarse-grid case, gamma error reached approximately **20× the true peak**. Rannacher time stepping suppresses the oscillation.

### Pathwise digital delta

A naive pathwise derivative of a digital payoff returns **exactly zero**, with zero estimated standard error, despite the true delta being nonzero. A smoothed-payoff estimator recovers the correct behavior.

These cases are included to demonstrate that **passing a numerical test is not enough; the test itself must target the relevant failure mode.**

---

## Figures

Generated analysis includes:

* `figures/mc_convergence.png` — Monte Carlo convergence and variance reduction
* `figures/pde_convergence.png` — PDE convergence behavior
* `figures/barrier_monitoring_bias.png` — discrete barrier-monitoring bias
* `figures/barrier_near_barrier.png` — BGK behavior near the barrier
* `figures/digital_pde_oscillation.png` — Crank–Nicolson digital-payoff oscillations
* `figures/digital_delta.png` — digital delta estimator comparison
* `figures/greeks_four_ways.png` — comparison of Greek estimation methods
* `figures/dg_error_vs_shock.png` — delta-gamma approximation error versus spot shock
* `figures/dg_pnl_curves.png` — approximate versus fully repriced P&L
* `figures/dg_error_heatmap.png` — portfolio approximation error across shocks
* `figures/dg_leg_errors.png` — leg-level contribution to approximation error

---

## Project structure

```text
derivatives-pricing-validation/
├── src/
│   ├── analytic.py
│   ├── mc_pricer.py
│   ├── pde_pricer.py
│   ├── exotics.py
│   ├── convergence.py
│   ├── greeks.py
│   └── plots.py
├── tests/
├── figures/
├── report.md
├── README.md
└── requirements.txt
```

 `src/` contains the pricing and analysis methods, while `tests/` contains the numerical validation and acceptance criteria.

---
```

For the detailed methodology, numerical results, see **[`report.md`](report.md)**.
