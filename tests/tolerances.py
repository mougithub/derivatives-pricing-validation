"""
PASS/FAIL TOLERANCES -- written BEFORE any test was run (Day 1).
Rule: change a value only with a dated, written reason in README.md "Change log".

Note on seeds: every random stream is seeded from SEED plus a test id, so all
results are reproducible. Do not hunt for "good" seeds.
"""

SEED = 20261005

# Reference contract used in most tests (continuous dividend yield q)
REF = dict(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2, q=0.02)

# ---------------- Analytic layer ----------------
PARITY_ABS = 1e-10              # C - P - (S e^{-qT} - K e^{-rT})
BOUNDS_SLACK = 1e-12            # float slack on no-arbitrage bounds
GREEKS_FD_REL = 1e-5            # analytic delta/gamma/vega vs central finite difference
LIMIT_ABS = 1e-3                # sigma->0 and T->0 limits (abs, price units)

BARRIER_CONTINUITY_ABS = 1e-9   # H<=K branch vs H>K branch at H=K
BARRIER_VANILLA_LIMIT_ABS = 1e-8  # H -> 0 recovers the vanilla call
BARRIER_NEAR_S_ABS = 1e-3       # H -> S gives ~0
BARRIER_MC_N_PATHS = 500_000    # independent cross-check of the closed form
BARRIER_MC_N_STEPS = 200        # (fine monitoring + BGK shift)
BARRIER_MC_Z_MAX = 4.0          # |MC - closed form| / SE

GEO_ASIAN_PARITY_ABS = 1e-10
GEO_ASIAN_CONT_LIMIT_REL = 1e-3  # discrete n=1e5 vs continuous (Kemna-Vorst) formula

# ---------------- Monte Carlo layer ----------------
MC_N_PATHS = 1_000_000
MC_Z_MAX = 3.0                  # analytic price within 3 SE

# Coverage: 93%-97% of seeds contain the analytic price.
# NOTE (written before running): with only 200 seeds the binomial SD of the
# observed coverage is 1.5 pts, so a CORRECT pricer would fail a 93-97% band
# about 14% of the time. We use 1000 seeds (SD 0.7 pts, false-fail ~0.3%).
COVERAGE_N_SEEDS = 1000
COVERAGE_N_PATHS = 20_000
COVERAGE_LOW, COVERAGE_HIGH = 0.93, 0.97

# Convergence: log-log slope of RMSE vs N in [-0.55, -0.45]
SLOPE_N_GRID = [1_000, 3_000, 10_000, 30_000, 100_000]
SLOPE_REPS = 500
SLOPE_LOW, SLOPE_HIGH = -0.55, -0.45

ANTITHETIC_SE_RATIO_MAX = 0.95  # antithetic SE / plain SE at equal N, ATM call

MC_ASIAN_N_FIX = 12             # monthly fixings
