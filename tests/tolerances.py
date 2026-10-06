"""
PASS/FAIL TOLERANCES.
change a value only with a dated, written reason in README.md "Change log".

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


# ================= PDE layer (written BEFORE the solver existed) =================
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


# ================= known-bad cases and Greeks (written BEFORE the code existed) =================
# ---- Barrier: discrete-monitoring bias of MC and its BGK fix; PDE vs closed form ----
BARRIER_CASES = [(100.0, 90.0), (90.0, 95.0)]    # (K, H): H<=K branch and H>K branch; S0 = 100
BARRIER_STEPS_GRID = [25, 50, 100, 200, 400]     # monitoring dates
BARRIER_BIAS_N_PATHS = 300_000
BARRIER_NAIVE_Z_MIN = 5.0          # naive MC must exceed the closed form by > 5 SE at 50 dates (bad case reproduced)
BARRIER_NAIVE_SLOPE_LOW, BARRIER_NAIVE_SLOPE_HIGH = -0.65, -0.35   # naive bias ~ n^(-1/2)
BARRIER_BGK_FRACTION = 0.2         # for n >= 50: |BGK error| <= 0.2*|naive bias| + 3 SE
BARRIER_PDE_SPOTS = [92.0, 95.0, 100.0, 105.0, 110.0]   # H=90, K=100
BARRIER_PDE_REL = 1e-3             # continuous-barrier PDE (M=N=800) vs closed form

# ---- Digital: CN oscillation near the discontinuity and Rannacher ----
# Errors on grid nodes with |ln(S/K)| <= DIGITAL_WINDOW, normalised by max |analytic value| in the window.
DIGITAL_WINDOW = 0.10
DIGITAL_CN_DELTA_MIN, DIGITAL_CN_GAMMA_MIN = 0.05, 0.20    # PRE-SET expectation: plain CN at M=N=800 at least this bad.
#                                                            REFUTED on first run (observed 3e-5 / 4e-5). No test uses these now.
DIGITAL_RAN_DELTA_MAX, DIGITAL_RAN_GAMMA_MAX = 0.01, 0.05  # CN + Rannacher must be at most this bad (pre-set, still used)
# --- POST-HOC additions (written AFTER the exploration in the README change log; these are not pre-registered) ---
DIGITAL_COARSE_N = 100             # coarse time grid at which plain CN does oscillate (M stays 800)
DIGITAL_CN_GAMMA_MIN_COARSE = 1.0  # plain CN gamma error at least equal to the peak true gamma (observed ~20)
DIGITAL_FINE_CN_MAX = 1e-3         # characterisation: at M=N=800 plain CN with a cell-averaged node is already clean
DIGITAL_PRICE_REL = 1e-3           # Rannacher digital price vs analytic, S0 in {95,100,105}

# ---- Greeks four ways (vanilla call, T=1): analytic | bump (CRN) | pathwise | PDE grid ----
GREEKS_N_PATHS = 1_000_000
GREEKS_SPOTS = [80.0, 90.0, 100.0, 110.0, 120.0]
GREEKS_Z_MAX = 4.0                 # MC estimate within 4 SE ...
GREEKS_BIAS_REL = 1e-3             # ... plus 1e-3 relative (finite-difference bias allowance; plan's bump criterion)
GREEKS_BUMP_REL = 0.01             # spot bump = 1% of S ; vol bump 0.005 absolute
GREEKS_VOL_BUMP = 0.005
PDE_VEGA_REL = 1e-3                # PDE vega (central bump, fixed grid) vs analytic
NEAR_EXPIRY_T = 0.005              # ~1.3 trading days
NEAR_EXPIRY_ADAPTIVE_H = 0.1       # adaptive spot bump = 0.1 * S * sigma * sqrt(T)
NEAR_EXPIRY_PDE_GAMMA_REL = 1e-2
# Known-bad: at NEAR_EXPIRY_T the FIXED 1% bump gamma must violate (4 SE + 1e-3 rel); the adaptive bump must pass.

# ---- Digital delta (cash-or-nothing, payout 1) ----
DIGITAL_GREEKS_SPOTS = [95.0, 100.0, 105.0]
DIGITAL_SMOOTH_EPS_REL = 0.01      # ramp half-width = 1% of S
DIGITAL_BIAS_REL = 1e-2            # smoothing / bump bias allowance (relative), on top of 4 SE
DIGITAL_PATHWISE_ZERO_ABS = 1e-12  # naive pathwise delta is exactly 0 (silent failure); analytic is not


# ================= delta-gamma(-vega) vs full repricing (written BEFORE the book code existed) =================
# Book (unit quantities): SHORT call K=110, SHORT put K=90, LONG down-and-out call K=100, H=85.  Base S0=100, REF market.
# Instantaneous shocks: spot S0*(1+s) and additive vol sigma+v; maturity fixed. Full repricing = closed forms
# (verified against MC and PDE on Days 1-3); a spot at or below H means the barrier is already hit (value 0).
BOOK_STRANGLE_K = (90.0, 110.0)       # (put strike, call strike)
BOOK_DOC_K, BOOK_DOC_H = 100.0, 85.0
BOOK_T_GRID = [1.0, 0.25, 0.05]
SPOT_SHOCKS = [0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30]   # applied with both signs
VOL_SHOCKS = [-0.10, -0.075, -0.05, -0.025, 0.0, 0.025, 0.05, 0.075, 0.10]    # vol points (0.05 = 5 vol pts)

# Book Greeks at the base point: closed-form barrier Greeks are not implemented, so the DOC Greeks are central
# differences of the closed form (h_S = 1e-3*S, h_sigma = 1e-4). They are checked against the PDE grid:
BOOK_GREEKS_PDE_DELTA_REL, BOOK_GREEKS_PDE_GAMMA_REL, BOOK_GREEKS_PDE_VEGA_REL = 1e-3, 1e-2, 1e-3
FULL_REPRICE_PDE_REL = 1e-3           # closed-form DOC vs PDE at shocked spot/vol with S >= 1.05*H (T=1)

# "Approximation unreliable" flag: |approx P&L - full P&L| / max(|full P&L|, DG_FLOOR_FRAC * gross base value) > DG_FLAG_REL.
# Gross base value = sum of |position values|; the floor stops the ratio exploding when the true P&L is near 0.
DG_FLAG_REL = 0.05
DG_FLOOR_FRAC = 0.01

# Expectations (pre-set):
DG_SLOPE_SHOCKS = [0.01, 0.02, 0.03, 0.05]          # small shocks used to fit the error-vs-shock slope
DG_SLOPE_STRANGLE = (2.7, 3.3)                      # smooth payoffs: delta-gamma error ~ |s|^3 (max over the two signs)
DG_SLOPE_BOOK = (2.5, 3.5)                          # book at T=1 (barrier far enough away at these shocks)
DG_RELIABLE_S = 0.03                                # T=1, v=0: delta-gamma NOT flagged for |s| <= 3%
# T=1, v=0: flagged at s=-0.20 (below the barrier); flagged at SOME upside shock <= +30%;
# the reliability frontier (smallest flagged |s|, either sign, step 0.5%) at T=0.05 is smaller than at T=1.
DG_FRONTIER_STEP = 0.005
DG_FRONTIER_MAX = 0.30
