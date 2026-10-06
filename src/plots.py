"""   (writes to figures/, prints the numbers quoted in the README)."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "tests")]
import tolerances as T
from black_scholes import bs_greeks
from exotics import down_and_out_call, digital_call_greeks
from mc_pricer import mc_down_and_out_call_pair
from pde_pricer import pde_price, pde_down_and_out_call, grid_greeks
import greeks as G
from greeks import Position

R = T.REF
FIG = os.path.join(HERE, "..", "figures")
A = dict(K=R["K"], T=R["T"], r=R["r"], sigma=R["sigma"], q=R["q"])


def fig_barrier_bias():
    K, H = T.BARRIER_CASES[0]
    cf = down_and_out_call(R["S"], K, H, R["T"], R["r"], R["sigma"], R["q"])
    ns, nb, ne, gb, ge = [], [], [], [], []
    for n in T.BARRIER_STEPS_GRID:
        a, b = mc_down_and_out_call_pair(R["S"], K, H, R["T"], R["r"], R["sigma"], R["q"], n_steps=n,
                                         n_paths=T.BARRIER_BIAS_N_PATHS, seed=[T.SEED, 7, int(K), int(H), n])
        ns.append(n); nb.append(a.price - cf); ne.append(a.se); gb.append(b.price - cf); ge.append(b.se)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.errorbar(ns, nb, yerr=[1.96 * e for e in ne], fmt="o-", capsize=3, label="naive discrete monitoring")
    ax.errorbar(ns, gb, yerr=[1.96 * e for e in ge], fmt="s-", capsize=3, label="with BGK barrier shift")
    ax.axhline(0, color="k", lw=0.8)
    n = np.array(ns, float); ax.plot(n, nb[0] * (n / n[0]) ** -0.5, "k--", lw=1, label="reference slope -1/2")
    ax.set_xscale("log"); ax.set_xlabel("monitoring dates n"); ax.set_ylabel("MC price - continuous closed form")
    ax.set_title(f"Down-and-out call (K={K:.0f}, H={H:.0f}): monitoring bias"); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{FIG}/barrier_monitoring_bias.png", dpi=150); plt.close(fig)


def fig_barrier_near():
    K, H, n = 100.0, 90.0, 52
    spots = np.round(np.arange(90.5, 110.01, 1.0), 2)
    cf, nv, bg, pd0, pd1 = [], [], [], [], []
    for S0 in spots:
        cf.append(down_and_out_call(S0, K, H, R["T"], R["r"], R["sigma"], R["q"]))
        a, b = mc_down_and_out_call_pair(S0, K, H, R["T"], R["r"], R["sigma"], R["q"], n_steps=n, n_paths=200_000,
                                         seed=[T.SEED, 8, int(S0 * 10)])
        nv.append(a.price); bg.append(b.price)
        pd0.append(pde_down_and_out_call(S0, K, H, R["T"], R["r"], R["sigma"], R["q"])["price"])
        pd1.append(pde_down_and_out_call(S0, K, H, R["T"], R["r"], R["sigma"], R["q"], rannacher=True)["price"])
    cf, nv, bg, pd0, pd1 = map(np.array, (cf, nv, bg, pd0, pd1))
    fig, axs = plt.subplots(2, 1, figsize=(7, 6.5), sharex=True, gridspec_kw=dict(height_ratios=[2, 1.3]))
    axs[0].plot(spots, cf, "k-", lw=2, label="closed form (continuous)")
    axs[0].plot(spots, nv, "o", ms=4, label=f"naive MC, {n} dates")
    axs[0].plot(spots, bg, "s", ms=4, label=f"BGK-corrected MC, {n} dates")
    axs[0].plot(spots, pd0, "--", label="PDE (CN)"); axs[0].plot(spots, pd1, ":", lw=2, label="PDE (CN + Rannacher)")
    axs[0].set_ylabel("price"); axs[0].legend(fontsize=8); axs[0].grid(True, alpha=0.3)
    axs[0].set_title("Down-and-out call near the barrier (H=90, K=100)")
    for y, l, m in [(nv - cf, "naive MC", "o"), (bg - cf, "BGK MC", "s"), (pd0 - cf, "PDE", "--")]:
        axs[1].plot(spots, y, m if m != "--" else "-", ms=4, label=l)
    axs[1].axhline(0, color="k", lw=0.8); axs[1].axvline(H * np.exp(0.5826 * R["sigma"] * np.sqrt(R["T"] / n)),
                                                           color="gray", ls=":", lw=1)
    axs[1].set_ylabel("error vs closed form"); axs[1].set_xlabel("spot"); axs[1].legend(fontsize=8); axs[1].grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{FIG}/barrier_near_barrier.png", dpi=150); plt.close(fig)
    print("near-barrier: BGK barrier sits at", round(90 * np.exp(0.5826 * R["sigma"] * np.sqrt(R["T"] / n)), 2),
          "| BGK-naive errors at S=91.5/95/100/110:",
          [(float(spots[i]), round(float(bg[i] - cf[i]), 3), round(float(nv[i] - cf[i]), 3)) for i in (0, 1, 5, 10, 19)])


def fig_digital_pde():
    K = R["K"]; Ex = (K, R["T"], R["r"], R["sigma"], R["q"])
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, key, title in [(axs[0], "delta", "digital delta"), (axs[1], "gamma", "digital gamma")]:
        Sg = np.linspace(K * np.exp(-0.1), K * np.exp(0.1), 400)
        ax.plot(Sg, digital_call_greeks(Sg, *Ex)[key], "k-", lw=2, label="analytic")
        for ran, N, st in [(False, T.DIGITAL_COARSE_N, "-"), (True, T.DIGITAL_COARSE_N, "--")]:
            res = pde_price("digital", K, *Ex, M=T.PDE_FINE_M, N=N, rannacher=ran)
            xi, d, g = grid_greeks(res["x"], res["V"]); w = np.abs(xi - np.log(K)) <= 0.1
            ax.plot(np.exp(xi[w]), (d if key == "delta" else g)[w], st, lw=1, label=f"PDE N={N}, " + ("CN + Rannacher" if ran else "plain CN"))
        ax.set_title(f"{title} near strike (M={T.PDE_FINE_M})"); ax.set_xlabel("spot"); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    axs[1].set_ylim(-0.0015, 0.0015)
    fig.tight_layout(); fig.savefig(f"{FIG}/digital_pde_oscillation.png", dpi=150); plt.close(fig)


def fig_digital_delta():
    spots = np.arange(90.0, 110.01, 2.0)
    ex = np.array([float(digital_call_greeks(S, **A)["delta"]) for S in spots])
    nz = [G.mc_digital_delta_pathwise_naive(S, **A, n_paths=200_000, seed=1) for S in spots]
    sm = [G.mc_digital_delta_pathwise_smoothed(S, **A, n_paths=1_000_000, seed=[T.SEED, 14, int(S)]) for S in spots]
    bu = [G.mc_digital_delta_bump(S, **A, n_paths=1_000_000, seed=[T.SEED, 15, int(S)]) for S in spots]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(spots, ex, "k-", lw=2, label="analytic")
    ax.plot(spots, [e.value for e in nz], "rx-", label="pathwise, naive (always 0)")
    ax.errorbar(spots, [e.value for e in sm], yerr=[1.96 * e.se for e in sm], fmt="o", capsize=2, label="pathwise, smoothed payoff")
    ax.errorbar(spots + 0.3, [e.value for e in bu], yerr=[1.96 * e.se for e in bu], fmt="s", capsize=2, label="bump-and-revalue (CRN)")
    ax.set_xlabel("spot"); ax.set_ylabel("digital delta"); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    ax.set_title("Digital call delta: pathwise fails, smoothing fixes it")
    fig.tight_layout(); fig.savefig(f"{FIG}/digital_delta.png", dpi=150); plt.close(fig)


def fig_greeks_four_ways():
    spots = np.arange(70.0, 130.01, 5.0)
    an = {k: np.array([float(bs_greeks(S, **A, kind="call")[k]) for S in spots]) for k in ("delta", "gamma", "vega")}
    bump = [G.mc_call_greeks_bump(S, **A, n_paths=T.GREEKS_N_PATHS, h_rel=T.GREEKS_BUMP_REL, vol_bump=T.GREEKS_VOL_BUMP,
                                  seed=[T.SEED, 10, int(S)]) for S in spots]
    pw = [G.mc_call_greeks_pathwise(S, **A, n_paths=T.GREEKS_N_PATHS, seed=[T.SEED, 11, int(S)]) for S in spots]
    pd = [G.pde_vanilla_greeks(S, **A, M=T.PDE_FINE_M, N=T.PDE_FINE_N) for S in spots]
    fig, axs = plt.subplots(2, 3, figsize=(13, 6.5), sharex=True)
    for j, k in enumerate(("delta", "gamma", "vega")):
        ax, ad = axs[0, j], axs[1, j]
        ax.plot(spots, an[k], "k-", lw=2, label="analytic")
        ax.plot(spots, [b[k].value for b in bump], "o", ms=4, color="C0", label="bump-and-revalue (CRN)")
        if k != "gamma":
            ax.plot(spots, [p[k].value for p in pw], "s", ms=4, color="C1", label="pathwise")
        ax.plot(spots, [p[k] for p in pd], "^", ms=4, color="C2", label="PDE grid")
        ax.set_title(k); ax.grid(True, alpha=0.3)
        ad.errorbar(spots, [b[k].value - a for b, a in zip(bump, an[k])], yerr=[1.96 * b[k].se for b in bump], fmt="o", color="C0", capsize=2, label="bump (95% CI)")
        if k != "gamma":
            ad.errorbar(spots + 0.5, [p[k].value - a for p, a in zip(pw, an[k])], yerr=[1.96 * p[k].se for p in pw], fmt="s", color="C1", capsize=2, label="pathwise (95% CI)")
        ad.plot(spots, [p[k] - a for p, a in zip(pd, an[k])], "^-", ms=4, color="C2", label="PDE grid")
        ad.axhline(0, color="k", lw=0.8); ad.set_xlabel("spot"); ad.set_ylabel("method - analytic"); ad.grid(True, alpha=0.3)
    axs[0, 0].legend(fontsize=8); axs[1, 0].legend(fontsize=7)
    axs[0, 1].text(0.5, 0.9, "no pathwise gamma for a call\n(integrand has zero 2nd derivative a.e.)", transform=axs[0, 1].transAxes, ha="center", fontsize=8)
    fig.suptitle("Greeks four ways, European call (T=1)"); fig.tight_layout()
    fig.savefig(f"{FIG}/greeks_four_ways.png", dpi=150); plt.close(fig)


# ======================= delta-gamma(-vega) vs full repricing =======================
PUT = Position("put", T.BOOK_STRANGLE_K[0], -1.0)
CALL = Position("call", T.BOOK_STRANGLE_K[1], -1.0)
DOC = Position("doc", T.BOOK_DOC_K, 1.0, T.BOOK_DOC_H)
BOOK, STRANGLE = (PUT, CALL, DOC), (PUT, CALL)
S0, MR, MS, MQ = R["S"], R["r"], R["sigma"], R["q"]


def _pnls(book, Tm, s, v=0.0, use_vega=True):
    g = G.book_greeks(book, S0, Tm, MR, MS, MQ)
    dS = s * S0
    return G.approx_pnl(g, dS, v, use_vega), G.full_pnl(book, S0, Tm, MR, MS, MQ, dS, v), G.gross_value(book, S0, Tm, MR, MS, MQ)


def fig_dg_error_vs_shock():
    ss = np.exp(np.linspace(np.log(0.005), np.log(0.30), 40))
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    series = [("strangle only, T=1", STRANGLE, 1.0)] + [(f"book, T={t:g}", BOOK, t) for t in T.BOOK_T_GRID]
    for lab, bk, t in series:
        err = [max(abs(np.subtract(*_pnls(bk, t, sg * s, use_vega=False)[:2])) for sg in (-1, 1)) for s in ss]
        ax.loglog(ss * 100, err, "-", label=lab)
    ref = ss ** 3 * (0.02 / 0.01 ** 3)
    ax.loglog(ss * 100, ref, "k--", lw=1, label="slope 3 reference")
    ax.set_xlabel("spot shock |s| (%)"); ax.set_ylabel("|delta-gamma P&L - full P&L|  (max of up / down)")
    ax.set_title("Delta-gamma error vs shock size"); ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{FIG}/dg_error_vs_shock.png", dpi=150); plt.close(fig)


def fig_dg_pnl_curves():
    ss = np.linspace(-0.30, 0.30, 121)
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2), sharex=True)
    for ax, t in zip(axs, T.BOOK_T_GRID):
        full = [_pnls(BOOK, t, s, use_vega=False)[1] for s in ss]
        dg = [_pnls(BOOK, t, s, use_vega=False)[0] for s in ss]
        ax.plot(ss * 100, full, "k-", lw=2, label="full repricing"); ax.plot(ss * 100, dg, "r--", label="delta-gamma")
        ax.axvline((T.BOOK_DOC_H / S0 - 1) * 100, color="gray", ls=":", label="barrier"); ax.set_title(f"book P&L, T={t:g}")
        ax.set_xlabel("spot shock (%)"); ax.grid(True, alpha=0.3)
    axs[0].set_ylabel("P&L"); axs[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{FIG}/dg_pnl_curves.png", dpi=150); plt.close(fig)


HEAT_CONTOUR = 2.0   # presentation choice made after seeing the data; NOT the pre-set 5% flag (see README)


def fig_dg_heatmap():
    ss = np.round(np.linspace(-0.30, 0.30, 25), 4)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
    stats = {}
    for ax, t in zip(axs, T.BOOK_T_GRID):
        Z = np.zeros((len(T.VOL_SHOCKS), len(ss)))
        for i, v in enumerate(T.VOL_SHOCKS):
            for j, s in enumerate(ss):
                a, f, gr = _pnls(BOOK, t, s, v)
                Z[i, j] = abs(a - f) / gr * 100     # fixed denominator: |error| as % of gross base value
        stats[t] = (Z <= HEAT_CONTOUR).mean()
        im = ax.imshow(np.minimum(Z, 10), origin="lower", aspect="auto", cmap="viridis_r", vmin=0, vmax=10,
                       extent=[ss[0] * 100 - 1.25, ss[-1] * 100 + 1.25, T.VOL_SHOCKS[0] * 100 - 1.25, T.VOL_SHOCKS[-1] * 100 + 1.25])
        ax.contour(ss * 100, np.array(T.VOL_SHOCKS) * 100, Z, levels=[HEAT_CONTOUR], colors="w", linewidths=1.5)
        ax.set_title(f"delta-gamma-vega error, T={t:g} (white = 2% of gross)"); ax.set_xlabel("spot shock (%)"); ax.set_ylabel("vol shock (vol pts)")
        fig.colorbar(im, ax=ax, label="|error| as % of gross base value (capped at 10)")
    fig.tight_layout(); fig.savefig(f"{FIG}/dg_error_heatmap.png", dpi=150); plt.close(fig)
    return stats


def fig_dg_leg_errors(t=0.25):
    shocks = [-0.20, -0.10, 0.10, 0.20]
    fig, ax = plt.subplots(figsize=(7, 4))
    w = 0.2
    for k, (lab, bk) in enumerate([("short put", (PUT,)), ("short call", (CALL,)), ("long DOC", (DOC,)), ("book (net)", BOOK)]):
        e = [np.subtract(*_pnls(bk, t, s, use_vega=False)[:2]) for s in shocks]
        ax.bar(np.arange(len(shocks)) + (k - 1.5) * w, e, w, label=lab, color=["C0", "C1", "C2", "k"][k])
    ax.set_xticks(range(len(shocks))); ax.set_xticklabels([f"{s:+.0%}" for s in shocks]); ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("delta-gamma minus full P&L"); ax.set_title(f"Leg errors offset in the book (T={t:g})"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{FIG}/dg_leg_errors.png", dpi=150); plt.close(fig)


def day4_numbers():
    print("vol-only shocks (s=0), delta-gamma-vega: abs error / metric%")
    for t in T.BOOK_T_GRID:
        row = []
        for v in (-0.10, -0.05, 0.05, 0.10):
            a, f, gr = _pnls(BOOK, t, 0.0, v)
            row.append(f"v={v:+.2f}: {a-f:+.3f}/{G.flag_metric(a,f,gr,T.DG_FLOOR_FRAC)*100:.1f}%")
        print(f"  T={t}:", " | ".join(row))
    print("vega term value: s=+-5%, v=+-5pts, metric% with vega vs without")
    for t in T.BOOK_T_GRID:
        a1, f1, g1 = _pnls(BOOK, t, 0.05, 0.05, True); a0, f0, g0 = _pnls(BOOK, t, 0.05, 0.05, False)
        print(f"  T={t}: with vega {G.flag_metric(a1,f1,g1,T.DG_FLOOR_FRAC)*100:.1f}%  without {G.flag_metric(a0,f0,g0,T.DG_FLOOR_FRAC)*100:.1f}%")
    print("frontier (down, up) with vega=False and v=0:")
    for t in T.BOOK_T_GRID:
        print("  T=", t, G.reliability_frontier(BOOK, S0, t, MR, MS, MQ, T.DG_FLOOR_FRAC, T.DG_FLAG_REL, T.DG_FRONTIER_STEP, T.DG_FRONTIER_MAX))
    print("strangle-only frontier T=1:", G.reliability_frontier(STRANGLE, S0, 1.0, MR, MS, MQ, T.DG_FLOOR_FRAC, T.DG_FLAG_REL, T.DG_FRONTIER_STEP, T.DG_FRONTIER_MAX))
    print("book at base: ", {t: {k: round(v, 4) for k, v in G.book_greeks(BOOK, S0, t, MR, MS, MQ).items()} for t in T.BOOK_T_GRID})


if __name__ == "__main__":
    for f in (fig_barrier_bias, fig_barrier_near, fig_digital_pde, fig_digital_delta, fig_greeks_four_ways):
        f(); print("done", f.__name__)
    for f in (fig_dg_error_vs_shock, fig_dg_pnl_curves, fig_dg_leg_errors):
        f(); print("done", f.__name__)
    print("fraction of (s,v) grid unflagged by T:", fig_dg_heatmap())
    day4_numbers()
