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


    
# ---------------- PDE convergence ----------------
def pde_order_study(r, sigma, q, T, K, p, space_M, space_N, time_N, time_M, S0=None, time_ref_N=2560):
    """Errors of the CN solver on the smooth payoff S^p (exact solution known).
    Returns relative errors at S0 for the space study and the time study, plus fitted orders."""
    from pde_pricer import pde_price
    S0 = K if S0 is None else S0
    cc = p * (r - q) + 0.5 * sigma**2 * p * (p - 1) - r
    exact = S0**p * np.exp(cc * T)
    err = lambda **kw: abs(pde_price("power", S0, K, T, r, sigma, q, p=p, **kw)["price"] - exact) / exact
    e_space = np.array([err(M=M, N=space_N) for M in space_M])
    # Time error is measured against a much finer-in-time solution on the SAME spatial grid, so the
    # (shared) spatial discretisation error cancels instead of forming a floor under the time error.
    price = lambda **kw: pde_price("power", S0, K, T, r, sigma, q, p=p, **kw)["price"]
    ref = price(M=time_M, N=time_ref_N)
    e_time = np.array([abs(price(M=time_M, N=N) - ref) / exact for N in time_N])
    # order in h: error ~ h^order with h ~ 1/M or 1/N  ->  slope of log(err) vs log(1/n)
    o_space = -loglog_slope(space_M, e_space)
    o_time = -loglog_slope(time_N, e_time)
    return dict(e_space=e_space, e_time=e_time, order_space=o_space, order_time=o_time)


def plot_pde_convergence(study, space_M, time_N, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for ax, n, e, o, lab in [(axs[0], space_M, study["e_space"], study["order_space"], "space points M"),
                             (axs[1], time_N, study["e_time"], study["order_time"], "time steps N")]:
        n = np.array(n, float)
        ax.loglog(n, e, "o-", label=f"observed order {o:.2f}")
        ax.loglog(n, e[0] * (n / n[0]) ** -2.0, "k--", lw=1, label="reference order 2")
        ax.set_xlabel(lab); ax.set_ylabel("relative error at S0"); ax.grid(True, which="both", alpha=0.3); ax.legend()
    axs[0].set_title("Crank-Nicolson, space (smooth payoff S^2)")
    axs[1].set_title("Crank-Nicolson, time (smooth payoff S^2)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)