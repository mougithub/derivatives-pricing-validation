"""MC convergence study: RMSE vs number of paths and fitted log-log slope."""
import numpy as np
from mc_pricer import mc_european
from black_scholes import bs_price


def mc_rmse_vs_n(params, n_grid, reps, antithetic, base_seed):
    exact = float(bs_price(**params))
    out = []
    for N in n_grid:
        errs = np.array([mc_european(**params, n_paths=N, antithetic=antithetic,
                                     seed=[base_seed, 2, N, i]).price - exact
                         for i in range(reps)])
        out.append(np.sqrt(np.mean(errs**2)))
    return np.array(out)


def loglog_slope(n_grid, rmse):
    return float(np.polyfit(np.log(n_grid), np.log(rmse), 1)[0])


def plot_mc_convergence(params, n_grid, reps, base_seed, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for anti, label in [(False, "plain"), (True, "antithetic")]:
        rmse = mc_rmse_vs_n(params, n_grid, reps, anti, base_seed)
        ax.loglog(n_grid, rmse, "o-", label=f"{label} (slope {loglog_slope(n_grid, rmse):.3f})")
    n = np.array(n_grid, float)
    ax.loglog(n, rmse[0] * (n / n[0]) ** -0.5, "k--", lw=1, label="reference slope -1/2")
    ax.set_xlabel("paths N"); ax.set_ylabel("RMSE vs analytic price")
    ax.set_title(f"MC error vs N, European call ({reps} seeds per point)")
    ax.legend(); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
