"""Monte Carlo pricers under exact GBM simulation, with optional antithetic variates.

Every function returns an MCResult with price, standard error and 95% CI.
With antithetic variates the SE is computed on the *pair averages*
(0.5*(f(z)+f(-z))), which are the i.i.d. units; using all 2m payoffs as if
independent would understate the SE.
"""
from dataclasses import dataclass
import numpy as np

Z95 = 1.959963984540054


@dataclass
class MCResult:
    price: float
    se: float
    n_paths: int
    antithetic: bool
    seed: object

    @property
    def ci(self):
        return (self.price - Z95 * self.se, self.price + Z95 * self.se)

    def contains(self, x):
        lo, hi = self.ci
        return lo <= x <= hi


def _summarize(units, disc, n_paths, antithetic, seed):
    units = disc * units
    m = units.size
    return MCResult(price=float(units.mean()), se=float(units.std(ddof=1) / np.sqrt(m)),
                    n_paths=n_paths, antithetic=antithetic, seed=seed)


def _payoff(x, K, kind):
    return np.maximum(x - K, 0.0) if kind == "call" else np.maximum(K - x, 0.0)


def mc_european(S, K, T, r, sigma, q=0.0, kind="call", n_paths=1_000_000,
                antithetic=True, seed=0):
    """seed may be an int or a list of ints (passed to numpy default_rng)."""
    rng = np.random.default_rng(seed)
    drift = (r - q - 0.5 * sigma**2) * T
    vol = sigma * np.sqrt(T)
    if antithetic:
        if n_paths % 2:
            raise ValueError("n_paths must be even with antithetic variates")
        z = rng.standard_normal(n_paths // 2)
        units = 0.5 * (_payoff(S * np.exp(drift + vol * z), K, kind)
                       + _payoff(S * np.exp(drift - vol * z), K, kind))
    else:
        z = rng.standard_normal(n_paths)
        units = _payoff(S * np.exp(drift + vol * z), K, kind)
    return _summarize(units, np.exp(-r * T), n_paths, antithetic, seed)


def _asian_payoff(z, S, K, T, r, sigma, q, kind, average):
    n = z.shape[1]
    dt = T / n
    logS = np.log(S) + np.cumsum((r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z, axis=1)
    avg = np.exp(logS.mean(axis=1)) if average == "geometric" else np.exp(logS).mean(axis=1)
    return _payoff(avg, K, kind)


def mc_asian(S, K, T, r, sigma, q=0.0, kind="call", n_fix=12, average="geometric",
             n_paths=500_000, antithetic=True, seed=0):
    """Discretely monitored Asian option, fixings at t_i = iT/n_fix, i=1..n_fix."""
    if average not in ("geometric", "arithmetic"):
        raise ValueError("average must be 'geometric' or 'arithmetic'")
    rng = np.random.default_rng(seed)
    if antithetic:
        if n_paths % 2:
            raise ValueError("n_paths must be even with antithetic variates")
        z = rng.standard_normal((n_paths // 2, n_fix))
        units = 0.5 * (_asian_payoff(z, S, K, T, r, sigma, q, kind, average)
                       + _asian_payoff(-z, S, K, T, r, sigma, q, kind, average))
    else:
        z = rng.standard_normal((n_paths, n_fix))
        units = _asian_payoff(z, S, K, T, r, sigma, q, kind, average)
    return _summarize(units, np.exp(-r * T), n_paths, antithetic, seed)
