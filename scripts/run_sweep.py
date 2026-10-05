#!/usr/bin/env python3
"""Minimal reproduction + round-1 experiment: grid over (simulated user x fixed assistance gamma).

For every user profile and every gamma, run N paired episodes on BlockPushMultimodal-v1
and record whether the block reached the user's goal (correct), the other goal (wrong,
i.e. the copilot overrode the user's intent) or neither (timeout).

Usage:
    python scripts/run_sweep.py                       # full default grid
    python scripts/run_sweep.py --episodes 20 --users expert noisy-0.6 --gammas 0 0.2 0.4   # quick check
Results are appended per condition to results/raw/*.csv, so the script can be resumed.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from sha_exp.users import DEFAULT_USERS

DEFAULT_GAMMAS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1.0]

_G = {}


def _init_worker():
    import torch

    torch.set_num_threads(1)
    from sha_exp import core

    _G["core"] = core
    _G["expert"] = core.load_expert()
    _G["diffusion"] = core.load_diffusion()
    _G["env"] = core.make_env()


def _run_condition(job):
    user_name, gamma, episodes, seed0, out_path = job
    from sha_exp.users import SimulatedUser, UserSpec

    core, env = _G["core"], _G["env"]
    user = SimulatedUser(UserSpec.parse(user_name), _G["expert"], env.action_space)
    rows = []
    t0 = time.time()
    for ep in range(episodes):
        core.seed_episode(env, user, seed0 + ep)
        r = core.run_episode(env, user, _G["diffusion"], gamma)
        rows.append({"user": user_name, "gamma": gamma, "episode": ep, **r})
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    return user_name, gamma, df["success"].mean(), (df["outcome"] == "wrong").mean(), time.time() - t0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--users", nargs="+", default=DEFAULT_USERS)
    p.add_argument("--gammas", nargs="+", type=float, default=DEFAULT_GAMMAS)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=10_000, help="episode i uses seed (seed + i)")
    p.add_argument("--workers", type=int, default=mp.cpu_count())
    p.add_argument("--out", type=Path, default=Path("results"))
    p.add_argument("--force", action="store_true", help="re-run conditions that already have results")
    args = p.parse_args()

    args.out = args.out.resolve()
    raw = args.out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    jobs = []
    for u in args.users:
        for g in args.gammas:
            path = raw / f"{u}__g{g:.2f}.csv"
            if path.exists() and not args.force:
                prev = pd.read_csv(path)
                if len(prev) >= args.episodes:
                    continue
            jobs.append((u, g, args.episodes, args.seed, path))
    print(f"{len(jobs)} conditions to run ({len(args.users)} users x {len(args.gammas)} gammas, {args.episodes} eps each)")

    with mp.get_context("spawn").Pool(args.workers, initializer=_init_worker) as pool:
        for u, g, succ, wrong, dt in pool.imap_unordered(_run_condition, jobs):
            print(f"  {u:>11s}  gamma={g:.2f}  success={succ:.2f}  wrong-goal={wrong:.2f}  ({dt:.0f}s)", flush=True)

    frames = [pd.read_csv(f) for f in sorted(raw.glob("*.csv"))]
    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv(args.out / "sweep_episodes.csv", index=False)
    print(f"wrote {args.out / 'sweep_episodes.csv'} ({len(all_df)} episodes)")


if __name__ == "__main__":
    main()
