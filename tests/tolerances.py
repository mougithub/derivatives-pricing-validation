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

# ================= : PDE layer  =================
# Solver: Crank-Nicolson in x = ln S on a uniform grid, Dirichlet boundaries, domain
# ln K +/- PDE_N_SD * sigma * sqrt(T). Price/Greeks at S0 by cubic-spline interpolation of grid values.
PDE_N_SD = 6.0
PDE_FINE_M, PDE_FINE_N = 800, 800
PDE_SPOTS = [80.0, 90.0, 100.0, 110.0, 120.0]

PDE_PRICE_REL = 1e-4            # vs analytic, vanilla call and put, fine grid (plan: < 1e-4)
PDE_PARITY_ABS = 1e-4           # C_pde - P_pde vs S e^{-qT} - K e^{-rT}
PDE_DELTA_REL = 1e-3            # grid delta vs analytic (plan: < 1e-3 away from singularities)
PDE_GAMMA_REL = 1e-2            # grid gamma vs analytic (looser: second derivative of a kinked payoff)
PDE_DOMAIN_ABS = 1e-6           # price change when the domain is widened 6 -> 9 sd at equal dx

# Observed order of convergence. Smooth payoff = S^2 (exact solution known, exact Dirichlet data).
# The vanilla call has a kink (at a grid node when S0=K), so it is NOT the smooth case.
PDE_ORDER_LOW, PDE_ORDER_HIGH = 1.85, 2.15   # smooth payoff, space and time separately
PDE_SPACE_M_GRID = [50, 100, 200, 400]       # with N = PDE_SPACE_N (time error negligible)
PDE_SPACE_N = 10_000
PDE_TIME_N_GRID = [10, 20, 40, 80, 160]      # with M = PDE_TIME_M (space error negligible)
PDE_TIME_M = 4_000
PDE_KINK_ORDER_MIN = 1.5                     # ATM vanilla call, M = N refined together: reported, floor only
PDE_TIME_REF_N = 2560                       # reference solution for the time-error study (same spatial grid)
