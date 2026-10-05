#!/usr/bin/env python3
"""Round-1 analysis: do different simulated users need different assistance strength?

Reads results/sweep_episodes.csv (from run_sweep.py) and writes
  results/summary.csv          success / wrong-goal / timeout per (user, gamma) with 95% CI
  results/best_gamma.csv       per-user optimal gamma, its bootstrap stability, near-optimal set
  results/headline.txt         personalized-oracle vs best-uniform gap (bootstrap CI, cross-fitted)
  results/fig_heatmap.png, fig_curves.png, fig_personal_vs_uniform.png

All gamma conditions share episode seeds, so bootstrap resamples *episode indices* jointly
across gammas (paired comparison).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

USER_ORDER = ["expert", "noisy", "laggy", "gauss"]


def user_sort_key(name: str):
    kind, _, lvl = name.partition("-")
    return (USER_ORDER.index(kind) if kind in USER_ORDER else 99, float(lvl or 0))


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return c - h, c + h


def success_matrix(df: pd.DataFrame, users, gammas) -> np.ndarray:
    """S[u, g, e] = success of user u at gamma g on episode e (episodes aligned by seed)."""
    piv = df.pivot_table(index=["user", "gamma"], columns="episode", values="success")
    n_ep = piv.shape[1]
    S = np.full((len(users), len(gammas), n_ep), np.nan)
    for i, u in enumerate(users):
        for j, g in enumerate(gammas):
            S[i, j] = piv.loc[(u, g)].values
    return S


def personalization_gap(S: np.ndarray, idx_sel: np.ndarray, idx_eval: np.ndarray):
    """Choose gamma on idx_sel episodes, evaluate on idx_eval episodes.
    Returns (mean personalized success, mean uniform success, per-user chosen gamma idx, uniform gamma idx)."""
    sel = S[:, :, idx_sel].mean(axis=2)  # (U, G)
    ev = S[:, :, idx_eval].mean(axis=2)
    g_user = sel.argmax(axis=1)
    g_unif = sel.mean(axis=0).argmax()
    personal = ev[np.arange(len(S)), g_user].mean()
    uniform = ev[:, g_unif].mean()
    return personal, uniform, g_user, g_unif


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=Path("results"))
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    out = args.results

    df = pd.read_csv(out / "sweep_episodes.csv")
    users = sorted(df["user"].unique(), key=user_sort_key)
    gammas = sorted(df["gamma"].unique())

    # ---------- 1. summary table ----------
    rows = []
    for (u, g), d in df.groupby(["user", "gamma"]):
        n, k = len(d), d["success"].sum()
        lo, hi = wilson(k, n)
        rows.append({
            "user": u, "gamma": g, "n": n,
            "success": k / n, "success_lo": lo, "success_hi": hi,
            "wrong_goal": (d["outcome"] == "wrong").mean(),
            "timeout": (d["outcome"] == "timeout").mean(),
            "ep_len": d["ep_len"].mean(),
            "action_diff": d["action_diff"].mean(),
        })
    summary = pd.DataFrame(rows)
    summary["_k"] = summary["user"].map(user_sort_key)
    summary = summary.sort_values(["_k", "gamma"]).drop(columns="_k")
    summary.to_csv(out / "summary.csv", index=False, float_format="%.3f")

    # ---------- 2. per-user optimal gamma ----------
    S = success_matrix(df, users, gammas)
    U, G, E = S.shape
    mean = S.mean(axis=2)
    boot_best = np.zeros((U, G))
    gaps_full = []
    for _ in range(args.n_boot):
        idx = rng.integers(0, E, E)
        m = S[:, :, idx].mean(axis=2)
        boot_best[np.arange(U), m.argmax(axis=1)] += 1
        gaps_full.append(m.max(axis=1).mean() - m.mean(axis=0).max())
    boot_best /= args.n_boot

    best_rows = []
    for i, u in enumerate(users):
        j = mean[i].argmax()
        # near-optimal set: gammas whose paired difference to the best has a 95% CI covering 0
        near = []
        for jj in range(G):
            diffs = S[i, j] - S[i, jj]
            bs = np.array([diffs[rng.integers(0, E, E)].mean() for _ in range(500)])
            if np.quantile(bs, 0.025) <= 0:
                near.append(gammas[jj])
        best_rows.append({
            "user": u,
            "best_gamma": gammas[j],
            "best_success": mean[i, j],
            "success_no_assist": mean[i, gammas.index(0.0)] if 0.0 in gammas else np.nan,
            "bootstrap_p_best": boot_best[i, j],
            "near_optimal_gammas": " ".join(f"{g:g}" for g in near),
        })
    best = pd.DataFrame(best_rows)

    # ---------- 3. headline: personalized vs uniform ----------
    g_unif = mean.mean(axis=0).argmax()
    best["success_at_uniform_best"] = mean[:, g_unif]
    best["gain_vs_uniform"] = best["best_success"] - best["success_at_uniform_best"]
    best.to_csv(out / "best_gamma.csv", index=False, float_format="%.3f")

    # In-sample oracle is optimistically biased (max over noisy estimates), so also report a
    # 2-fold cross-fitted version: choose gamma on half of the episodes, evaluate on the other half.
    cf = []
    for _ in range(args.n_boot // 4):
        perm = rng.permutation(E)
        a, b = perm[: E // 2], perm[E // 2:]
        p1, u1, _, _ = personalization_gap(S, a, b)
        p2, u2, _, _ = personalization_gap(S, b, a)
        cf.append(((p1 - u1) + (p2 - u2)) / 2)
    cf = np.array(cf)
    gaps_full = np.array(gaps_full)

    lines = [
        f"users: {U}, gammas: {G}, episodes per condition: {E}",
        f"best uniform gamma: {gammas[g_unif]:g}  -> mean success {mean[:, g_unif].mean():.3f}",
        f"no assistance (gamma=0) mean success: {mean[:, gammas.index(0.0)].mean():.3f}" if 0.0 in gammas else "",
        f"personalized oracle mean success (in-sample): {mean.max(axis=1).mean():.3f}",
        f"gap personalized - uniform (in-sample): {mean.max(axis=1).mean() - mean[:, g_unif].mean():+.3f}  "
        f"95% CI [{np.quantile(gaps_full, .025):+.3f}, {np.quantile(gaps_full, .975):+.3f}]",
        f"gap personalized - uniform (2-fold cross-fitted, unbiased): {cf.mean():+.3f}  "
        f"95% interval [{np.quantile(cf, .025):+.3f}, {np.quantile(cf, .975):+.3f}]",
        f"distinct per-user best gammas: {sorted(set(best['best_gamma']))}",
        "",
        best.to_string(index=False, float_format=lambda x: f"{x:.3f}"),
    ]
    text = "\n".join(l for l in lines if l is not None)
    (out / "headline.txt").write_text(text + "\n")
    print(text)

    # ---------- 4. figures ----------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, (key, title) in zip(axes, [("success", "Correct-goal success"), ("wrong_goal", "Wrong-goal rate (intent overridden)")]):
        M = summary.pivot(index="user", columns="gamma", values=key).loc[users]
        im = ax.imshow(M.values, aspect="auto", cmap="viridis" if key == "success" else "magma", vmin=0, vmax=1)
        ax.set_xticks(range(G), [f"{g:g}" for g in gammas])
        ax.set_yticks(range(U), users)
        ax.set_xlabel("assistance strength  gamma (forward-diffusion ratio)")
        ax.set_title(title)
        for i in range(U):
            for j in range(G):
                ax.text(j, i, f"{M.values[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if M.values[i, j] < 0.5 else "black")
            if key == "success":
                ax.add_patch(plt.Rectangle((mean[i].argmax() - .5, i - .5), 1, 1, fill=False, ec="red", lw=2))
        fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(out / "fig_heatmap.png", dpi=150)

    kinds = [k for k in USER_ORDER if any(u.split("-")[0] == k for u in users)]
    fig, axes = plt.subplots(2, len(kinds), figsize=(4 * len(kinds), 7), sharex=True, sharey=True)
    axes = np.atleast_2d(axes).reshape(2, len(kinds))
    for c, kind in enumerate(kinds):
        for u in [u for u in users if u.split("-")[0] == kind]:
            d = summary[summary["user"] == u]
            l, = axes[0, c].plot(d["gamma"], d["success"], "o-", label=u, ms=4)
            axes[0, c].fill_between(d["gamma"], d["success_lo"], d["success_hi"], color=l.get_color(), alpha=.15)
            axes[1, c].plot(d["gamma"], d["wrong_goal"], "o-", ms=4, color=l.get_color())
        axes[0, c].set_title(kind)
        axes[0, c].legend(fontsize=8)
        axes[1, c].set_xlabel("gamma")
    axes[0, 0].set_ylabel("correct-goal success")
    axes[1, 0].set_ylabel("wrong-goal rate")
    fig.tight_layout()
    fig.savefig(out / "fig_curves.png", dpi=150)

    fig, ax = plt.subplots(figsize=(max(8, U * 0.8), 4))
    x = np.arange(U)
    ax.bar(x - .27, best["success_no_assist"], .27, label="no assist (gamma=0)", color="#bbbbbb")
    ax.bar(x, best["success_at_uniform_best"], .27, label=f"best uniform (gamma={gammas[g_unif]:g})", color="#4c72b0")
    ax.bar(x + .27, best["best_success"], .27, label="per-user best (oracle)", color="#dd8452")
    for i in range(U):
        ax.text(x[i] + .27, best["best_success"][i] + .01, f"{best['best_gamma'][i]:g}", ha="center", fontsize=7)
    ax.set_xticks(x, users, rotation=30)
    ax.set_ylabel("correct-goal success")
    ax.set_ylim(0, 1.08)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Personalized vs uniform assistance (number above bar = per-user best gamma)")
    fig.tight_layout()
    fig.savefig(out / "fig_personal_vs_uniform.png", dpi=150)
    print(f"figures written to {out}/")


if __name__ == "__main__":
    main()
