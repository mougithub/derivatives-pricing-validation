"""Crank-Nicolson finite-difference solver for the Black-Scholes PDE in x = ln S.

With tau = T - t:   V_tau = 0.5 s^2 V_xx + (r - q - 0.5 s^2) V_x - r V.
Uniform grid x_j = x_min + j dx, j = 0..M, constant coefficients, Dirichlet boundaries
(time-dependent, supplied per payoff). Each step solves one tridiagonal system.

Boundary conditions (documented per payoff; S_min = e^{x_min}, S_max = e^{x_max}):
  call  : V(S_min) = 0              V(S_max) = S_max e^{-q tau} - K e^{-r tau}
  put   : V(S_min) = K e^{-r tau} - S_min e^{-q tau}      V(S_max) = 0
  digital : cash-or-nothing call, payoff 1{S>K} with value 0.5 AT the strike node (cell average, grid centred
          on ln K). V(S_min) = 0, V(S_max) = e^{-r tau}.
  down-and-out call (pde_down_and_out_call): grid starts AT the barrier, V(H) = 0 (continuous monitoring, no rebate),
          V(S_max) = S_max e^{-q tau} - K e^{-r tau}.
  power : V = S^p e^{c tau}, c = p(r-q) + 0.5 s^2 p(p-1) - r  (exact solution; used to
          measure the observed order on a SMOOTH payoff)
These are asymptotically exact; the truncation error is tested by widening the domain.

rannacher=True replaces the first two CN steps by four half-size fully implicit steps,
which damps the high-frequency error CN leaves after a non-smooth payoff (used on Day 3).
"""
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.linalg import solve_banded


def solve_bs_pde(payoff, lower_bc, upper_bc, T, r, sigma, q, x_min, x_max, M, N, rannacher=False):
    """Return (x, V) at tau = T. payoff(S) is vectorised; bc(tau) return floats."""
    x = np.linspace(x_min, x_max, M + 1)
    dx = x[1] - x[0]
    mu = r - q - 0.5 * sigma**2
    a = 0.5 * sigma**2 / dx**2 - mu / (2 * dx)      # coefficient of V_{j-1}
    b = -sigma**2 / dx**2 - r                        # coefficient of V_j
    c = 0.5 * sigma**2 / dx**2 + mu / (2 * dx)      # coefficient of V_{j+1}
    V = np.asarray(payoff(np.exp(x)), dtype=float).copy()

    if rannacher:
        if N < 2:
            raise ValueError("rannacher needs N >= 2")
        schedule = [(T / N / 2, 1.0)] * 4 + [(T / N, 0.5)] * (N - 2)
    else:
        schedule = [(T / N, 0.5)] * N

    cache, tau = {}, 0.0
    for dt, theta in schedule:
        if (dt, theta) not in cache:
            ab = np.zeros((3, M - 1))
            ab[0, 1:] = -theta * dt * c
            ab[1, :] = 1.0 - theta * dt * b
            ab[2, :-1] = -theta * dt * a
            cache[(dt, theta)] = ab
        ab = cache[(dt, theta)]
        rhs = V[1:-1] + (1.0 - theta) * dt * (a * V[:-2] + b * V[1:-1] + c * V[2:])
        tau += dt
        lo, up = lower_bc(tau), upper_bc(tau)
        rhs[0] += theta * dt * a * lo                # known boundary values at the new time level
        rhs[-1] += theta * dt * c * up
        V[1:-1] = solve_banded((1, 1), ab, rhs)
        V[0], V[-1] = lo, up
    return x, V


def grid_greeks(x, V):
    """Delta and gamma at interior nodes from central differences in x."""
    dx = x[1] - x[0]
    Vx = (V[2:] - V[:-2]) / (2 * dx)
    Vxx = (V[2:] - 2 * V[1:-1] + V[:-2]) / dx**2
    S = np.exp(x[1:-1])
    return x[1:-1], Vx / S, (Vxx - Vx) / S**2          # d/dS = (1/S) d/dx ; d2/dS2 = (V_xx - V_x)/S^2


def _contract(kind, K, T, r, sigma, q, p):
    if kind == "call":
        return (lambda S: np.maximum(S - K, 0.0),
                lambda tau, Smin, Smax: 0.0,
                lambda tau, Smin, Smax: Smax * np.exp(-q * tau) - K * np.exp(-r * tau))
    if kind == "put":
        return (lambda S: np.maximum(K - S, 0.0),
                lambda tau, Smin, Smax: K * np.exp(-r * tau) - Smin * np.exp(-q * tau),
                lambda tau, Smin, Smax: 0.0)
    if kind == "power":
        cc = p * (r - q) + 0.5 * sigma**2 * p * (p - 1) - r
        return (lambda S: S**p,
                lambda tau, Smin, Smax: Smin**p * np.exp(cc * tau),
                lambda tau, Smin, Smax: Smax**p * np.exp(cc * tau))
    if kind == "digital":
        def pay(S):
            return np.where(np.isclose(S, K, rtol=1e-9), 0.5, (S > K).astype(float))
        return (pay, lambda tau, Smin, Smax: 0.0, lambda tau, Smin, Smax: np.exp(-r * tau))
    raise ValueError("kind must be 'call', 'put', 'digital' or 'power'")


def pde_price(kind, S0, K, T, r, sigma, q=0.0, M=800, N=800, n_sd=6.0, rannacher=False, p=2.0,
              half_width=None):
    """Price, delta, gamma at S0. Grid is centred on ln K (so a strike kink sits on a node
    when M is even) and spans ln K +/- n_sd*sigma*sqrt(T)."""
    half = n_sd * sigma * np.sqrt(T) if half_width is None else half_width   # override keeps dx fixed under a vol bump
    x_min, x_max = np.log(K) - half, np.log(K) + half
    x0 = np.log(S0)
    dx = (x_max - x_min) / M
    if not (x_min + 2 * dx < x0 < x_max - 2 * dx):
        raise ValueError("S0 too close to / outside the grid; increase n_sd")
    payoff, lo_f, up_f = _contract(kind, K, T, r, sigma, q, p)
    Smin, Smax = np.exp(x_min), np.exp(x_max)
    x, V = solve_bs_pde(payoff, lambda t: lo_f(t, Smin, Smax), lambda t: up_f(t, Smin, Smax),
                        T, r, sigma, q, x_min, x_max, M, N, rannacher)
    xi, delta, gamma = grid_greeks(x, V)
    return dict(price=float(CubicSpline(x, V)(x0)),
                delta=float(CubicSpline(xi, delta)(x0)),
                gamma=float(CubicSpline(xi, gamma)(x0)),
                x=x, V=V, dx=dx)


def pde_down_and_out_call(S0, K, H, T, r, sigma, q=0.0, M=800, N=800, n_sd=6.0, rannacher=False, x_max=None):
    """Continuously monitored down-and-out call. Grid runs from the barrier (a node) to
    ln K + n_sd sigma sqrt(T); the strike is generally NOT a node here."""
    x_min = np.log(H)
    x_max = max(np.log(K), np.log(S0)) + n_sd * sigma * np.sqrt(T) if x_max is None else x_max
    dx = (x_max - x_min) / M
    x0 = np.log(S0)
    if not (x_min + dx < x0 < x_max - 2 * dx):
        raise ValueError("S0 must lie strictly between the barrier and the upper boundary")
    Smax = np.exp(x_max)
    x, V = solve_bs_pde(lambda S: np.maximum(S - K, 0.0), lambda t: 0.0,
                        lambda t: Smax * np.exp(-q * t) - K * np.exp(-r * t),
                        T, r, sigma, q, x_min, x_max, M, N, rannacher)
    xi, delta, gamma = grid_greeks(x, V)
    return dict(price=float(CubicSpline(x, V)(x0)), delta=float(CubicSpline(xi, delta)(x0)),
                gamma=float(CubicSpline(xi, gamma)(x0)), x=x, V=V, dx=dx, x_max=x_max)
