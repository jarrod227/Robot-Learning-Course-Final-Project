#!/usr/bin/env python3
"""Toy illustration of diffusion-based assistance on a 1-D continuous action.

Expert actions form two continuous bumps (+0.5 = goal A, -0.5 = goal B); the user
sends 0.2. Uses the same beta schedule as the BlockPush checkpoint and the exact
(optimal) denoiser for a Gaussian mixture, i.e. what a perfectly trained network outputs.
Each reverse step is written in its intuitive form:
    x_{t-1} = c1 * x0_hat(x_t) + c2 * x_t + small noise
Output: docs/denoise_explained.png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "Noto Sans CJK SC", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
rng = np.random.default_rng(1)

T = 50
b = 1 / (1 + np.exp(-np.linspace(-6, 6, T))) * (0.26 - 1e-4) + 1e-4
a = 1 - b
ab = np.cumprod(a)
abp = np.concatenate([[1], ab[:-1]])
mus, sd = np.array([0.5, -0.5]), 0.15


def x0hat(x, i):
    """Posterior mean of the clean action given noisy x at level i (two-Gaussian mixture)."""
    A = ab[i]
    v = A * sd**2 + 1 - A
    logp = -(x[:, None] - np.sqrt(A) * mus) ** 2 / (2 * v)
    w = np.exp(logp - logp.max(1, keepdims=True))
    w /= w.sum(1, keepdims=True)
    x0 = mus + np.sqrt(A) * sd**2 / v * (x[:, None] - np.sqrt(A) * mus)
    return (w * x0).sum(1)


def step(x, i):
    c1 = np.sqrt(abp[i]) * b[i] / (1 - ab[i])
    c2 = np.sqrt(a[i]) * (1 - abp[i]) / (1 - ab[i])
    var = b[i] * (1 - abp[i]) / (1 - ab[i])
    return c1 * x0hat(x, i) + c2 * x + np.sqrt(var) * rng.standard_normal(x.shape)


fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
xs = np.linspace(-1.2, 1.2, 400)
dens = np.exp(-(xs - 0.5) ** 2 / (2 * sd**2)) + np.exp(-(xs + 0.5) ** 2 / (2 * sd**2))
ax[0].fill_between(xs, dens, alpha=0.3, color="C2")
ax[0].plot(xs, dens, "C2")
ax[0].axvline(0.2, color="C3", lw=2)
ax[0].text(0.22, 1.02, "用户动作 0.2", color="C3")
ax[0].set_title("① 专家动作的分布（连续，两团）")
ax[0].set_xlabel("动作")
ax[0].set_yticks([])
ax[0].text(0.5, 0.5, "目标A", ha="center")
ax[0].text(-0.5, 0.5, "目标B", ha="center")

for i, c in [(4, "C0"), (12, "C1"), (30, "C4")]:
    ax[1].plot(xs, x0hat(xs, i), c, label=f"第 {i} 级噪声")
ax[1].plot(xs, xs, "k:", lw=1, label="y = x（不修改）")
ax[1].set_title("② 网络的判断：看到带噪动作 x，\n它猜干净动作是多少")
ax[1].set_xlabel("看到的带噪动作 x")
ax[1].set_ylabel("猜的干净动作")
ax[1].legend(fontsize=8)
ax[1].set_ylim(-1, 1)
for y in (0.5, -0.5):
    ax[1].axhline(y, color="C2", lw=0.5)

for g, c in [(0.25, "C1"), (0.7, "C4")]:
    k = int(49 * g)
    x = np.sqrt(ab[k]) * 0.2 + np.sqrt(1 - ab[k]) * rng.standard_normal(25)
    tr = [x.copy()]
    for i in reversed(range(k)):
        x = step(x, i)
        tr.append(x.copy())
    tr = np.array(tr)
    steps = np.arange(-len(tr) + 1, 1)
    for j in range(tr.shape[1]):
        ax[2].plot(steps, tr[:, j], c, alpha=0.45, lw=1)
    ax[2].plot([], [], c, label=f"γ={g}（k={k}）：{(tr[-1] > 0).mean() * 100:.0f}% 到目标A")
ax[2].axhline(0.2, color="C3", ls="--", lw=1)
for y in (0.5, -0.5):
    ax[2].axhline(y, color="C2", lw=0.5)
ax[2].set_title("③ 从用户动作 0.2 出发：加噪后一步步去噪（各 25 次）")
ax[2].set_xlabel("去噪步（0 = 结束）")
ax[2].set_ylabel("动作")
ax[2].legend(fontsize=8, loc="lower left")

fig.tight_layout()
out = Path(__file__).resolve().parent.parent / "docs" / "denoise_explained.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=140)
print(f"wrote {out}")
