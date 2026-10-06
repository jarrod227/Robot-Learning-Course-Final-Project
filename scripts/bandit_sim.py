#!/usr/bin/env python3
"""Follow-up topic, preliminary: choose assistance strength per episode under limited feedback.

Each simulated user is a multi-armed bandit whose arms are the gamma grid; pulling arm g
= running one episode at gamma g and observing binary feedback (did the block reach my goal?).
To keep this cheap, episodes are *replayed* from the sweep's per-episode outcomes
(sampled with replacement from results/sweep_episodes.csv), so no simulator is needed.

Strategies (all get the same per-user budget of B episodes):
  uniform      fixed gamma chosen as the best average over the *other* users (leave-one-user-out)
  eps-greedy   epsilon = 0.1, ties broken randomly
  ucb1         UCB1
  ts           Thompson sampling, Beta(1,1) prior
  ts-prior     Thompson sampling with a population prior from the other users
               (Beta pseudo-counts = prior_strength * leave-one-out mean success)
  oracle       per-user best gamma (upper bound, not achievable)

Metrics as a function of budget B:
  cumulative success  mean success over the B calibration episodes (what the user experiences)
  final success       true success (full-data mean) of the gamma the strategy would recommend after B
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STRATEGIES = ["uniform", "eps-greedy", "ucb1", "ts", "ts-prior", "oracle"]


def run_one(strategy, pools, true_mean, loo_mean, B, rng, prior_strength=4.0):
    """pools[g] = array of 0/1 outcomes for this user at arm g. Returns (successes[B], recommended arm)."""
    G = len(pools)
    n = np.zeros(G)
    s = np.zeros(G)
    out = np.zeros(B)
    if strategy == "ts-prior":
        a0 = 1 + prior_strength * loo_mean
        b0 = 1 + prior_strength * (1 - loo_mean)
    else:
        a0 = b0 = np.ones(G)
    for t in range(B):
        if strategy == "uniform":
            g = int(loo_mean.argmax())
        elif strategy == "oracle":
            g = int(true_mean.argmax())
        elif strategy == "eps-greedy":
            if rng.random() < 0.1 or n.sum() == 0:
                g = int(rng.integers(G))
            else:
                est = np.where(n > 0, s / np.maximum(n, 1), -1)
                g = int(rng.choice(np.flatnonzero(est == est.max())))
        elif strategy == "ucb1":
            if (n == 0).any():
                g = int(rng.choice(np.flatnonzero(n == 0)))
            else:
                g = int((s / n + np.sqrt(2 * np.log(t + 1) / n)).argmax())
        elif strategy in ("ts", "ts-prior"):
            g = int(rng.beta(a0 + s, b0 + n - s).argmax())
        else:
            raise ValueError(strategy)
        r = pools[g][rng.integers(len(pools[g]))]
        n[g] += 1
        s[g] += r
        out[t] = r

    # recommendation after the budget: posterior/empirical mean (fixed strategies keep their arm)
    if strategy == "uniform":
        rec = int(loo_mean.argmax())
    elif strategy == "oracle":
        rec = int(true_mean.argmax())
    else:
        post = (a0 + s) / (a0 + b0 + n)
        rec = int(post.argmax())
    return out, rec


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", type=Path, default=Path("results"))
    p.add_argument("--budgets", nargs="+", type=int, default=[5, 10, 20, 40, 80])
    p.add_argument("--runs", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)

    df = pd.read_csv(args.results / "sweep_episodes.csv")
    users = sorted(df["user"].unique())
    gammas = sorted(df["gamma"].unique())
    pools = {u: [df[(df.user == u) & (df.gamma == g)]["success"].values for g in gammas] for u in users}
    true_mean = {u: np.array([pl.mean() for pl in pools[u]]) for u in users}

    rows = []
    for u in users:
        loo = np.mean([true_mean[v] for v in users if v != u], axis=0)
        for strat in STRATEGIES:
            for B in args.budgets:
                cum, fin = [], []
                for _ in range(args.runs):
                    out, rec = run_one(strat, pools[u], true_mean[u], loo, B, rng)
                    cum.append(out.mean())
                    fin.append(true_mean[u][rec])
                rows.append({"user": u, "strategy": strat, "budget": B,
                             "cumulative_success": np.mean(cum), "final_success": np.mean(fin)})
    res = pd.DataFrame(rows)
    res.to_csv(args.results / "bandit_per_user.csv", index=False, float_format="%.4f")
    agg = res.groupby(["strategy", "budget"])[["cumulative_success", "final_success"]].mean().reset_index()
    agg.to_csv(args.results / "bandit_summary.csv", index=False, float_format="%.4f")
    print(agg.pivot(index="strategy", columns="budget", values="cumulative_success").loc[STRATEGIES]
          .to_string(float_format=lambda x: f"{x:.3f}", header=True).replace("budget", "cumulative success | budget"))
    print()
    print(agg.pivot(index="strategy", columns="budget", values="final_success").loc[STRATEGIES]
          .to_string(float_format=lambda x: f"{x:.3f}").replace("budget", "final success      | budget"))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for strat in STRATEGIES:
        d = agg[agg.strategy == strat]
        ls = "--" if strat in ("uniform", "oracle") else "-"
        axes[0].plot(d.budget, d.cumulative_success, ls, marker="o", label=strat)
        axes[1].plot(d.budget, d.final_success, ls, marker="o", label=strat)
    axes[0].set_title("Mean success during calibration episodes")
    axes[1].set_title("True success of recommended gamma after budget")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks(args.budgets, [str(b) for b in args.budgets])
        ax.set_xlabel("per-user episode budget B")
    axes[0].set_ylabel("success (avg over users)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.results / "fig_bandit.png", dpi=150)


if __name__ == "__main__":
    main()
