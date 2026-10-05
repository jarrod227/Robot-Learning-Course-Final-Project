# Robot-Learning-Course-Final-Project

## A 同学：Shared Autonomy，从固定辅助强度走向个性化辅助

| | |
|---|---|
| **复现论文** | Yoneda et al., *To the Noise and Back: Diffusion for Shared Autonomy*, RSS 2023（[arXiv:2302.12244](https://arxiv.org/abs/2302.12244)） |
| **官方实现** | [ripl/diffusion-for-shared-autonomy](https://github.com/ripl/diffusion-for-shared-autonomy)，提供 demonstrations、预训练扩散模型、SAC 专家和模拟用户评测 |
| **最小复现** | 用 `BlockPushMultimodal-v1`，比较无辅助和不同固定辅助强度 γ 下，各模拟用户（不同噪声）的任务表现和用户目标保持情况 |
| **后续选题** | 有限反馈下，个性化辅助能否优于统一辅助强度？先做每回合选择辅助强度，在同样的预算下与简单的单用户校准做对比 |
| **第一轮要回答的问题** | 不同模拟用户是否需要不同的辅助强度？如果最优强度几乎相同，个性化的空间就很有限 |

---

### 1. 方法回顾

扩散模型 $p_\theta(a \mid s)$ 在专家 demonstrations 上训练。BlockPush 的数据被镜像翻转过，所以两个目标都有，动作分布是双峰的。
每一步，用户给出动作 $a_h$，copilot 做以下几步：

1. 把 $a_h$ **前向扩散** $k=\lfloor (T-1)\,\gamma \rfloor$ 步（$T=50$），得到带噪的 $a_k$；
2. 以当前观测 $s$ 为硬条件，从 $a_k$ **反向去噪** $k$ 步，得到执行动作 $a$。

$\gamma\in[0,1]$（论文中的 forward-diffusion ratio）就是**辅助强度**：

- $\gamma=0$：直接执行用户动作，不提供辅助；
- $\gamma\to1$：几乎从纯噪声开始采样，执行的是专家分布中的动作，但不一定朝向用户想去的目标。

BlockPush 场景里有两个目标区（`target` 与 `target2`），用户的意图是 `target`。观测里不含目标信息，所以 γ 太大时 copilot 可能把方块推到另一个目标，也就是**用户意图被覆盖**。这个环境因此能同时度量"任务完成"和"目标保持"。

### 2. 实验设计

**模拟用户**（`sha_exp/users.py`）。每个用户都是同一个 SAC 专家（推向 `target`）加上一种噪声：

| 用户 | 含义 | 等级 |
|---|---|---|
| `expert` | 不加噪声 | – |
| `noisy-ε` | 以概率 ε 输出均匀随机动作（等同论文中的 NoisyActor） | 0.2 / 0.4 / 0.6 / 0.8 |
| `laggy-p` | 以概率 p 重复上一步动作（等同论文中的 LaggyActor） | 0.3 / 0.5 / 0.7 / 0.85 |
| `gauss-σ` | 专家动作加高斯噪声后截断（额外增加的一类） | 0.5 / 1 |

**辅助强度网格**：γ ∈ {0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1.0}

**评测**：每个（用户, γ）跑 200 个 episode，每个最多 100 步。每个 episode 记一种结果：

- `correct`：方块进入用户目标，即成功；
- `wrong`：方块进入另一个目标，意图被覆盖；
- `timeout`：在步数上限内没有到达任何目标。

**配对设计（common random numbers）**：第 i 个 episode 在所有 γ 下使用同一个种子，因此初始场景、用户噪声序列和扩散噪声都相同。不同 γ 之间的比较是配对的，方差更小。bootstrap 时也按 episode 联合重采样。

**判断"个性化空间"的指标**（`scripts/analyze.py`）：

- 每个用户的最优 γ*，以及它在 bootstrap 中被选为最优的概率；
- 与最优值在统计上无法区分的 γ 集合（配对差的 95% CI 覆盖 0）；
- **个性化 oracle 与最优统一 γ 的平均成功率差**。in-sample 的 max 有乐观偏差，所以另外报告 2-fold cross-fitting 的结果：一半 episode 用来选 γ，另一半用来评估。

### 3. 第一轮结果

> RESULTS_PLACEHOLDER

### 4. 后续：有限反馈下的每回合辅助强度选择（初步）

`scripts/bandit_sim.py` 把每个用户建模成多臂老虎机：臂是 γ 网格，拉一次臂就是跑一个 episode，并拿到二值反馈（是否到达我的目标）。为了省算力，episode 结果直接从上面 sweep 的逐 episode 数据中**有放回重放**，不需要重新跑仿真。比较的策略：

- `uniform`：统一强度，取除该用户外其他用户平均最优的 γ（leave-one-user-out）；
- `eps-greedy` / `ucb1` / `ts`：单用户从零开始校准；
- `ts-prior`：用其他用户的数据作为先验的 Thompson sampling；
- `oracle`：该用户的真实最优 γ，作为上界。

> BANDIT_PLACEHOLDER

### 5. 运行方式

```bash
bash setup.sh                    # 克隆官方仓库（固定 commit，含预训练模型）并安装依赖（CPU 即可）
source .venv/bin/activate

python scripts/run_sweep.py      # 11 用户 × 11 γ × 200 episodes；4 核 CPU 约 1 小时，可断点续跑
python scripts/analyze.py        # 汇总表 + 图 + headline
python scripts/bandit_sim.py     # 后续选题的离线 bandit 模拟
```

快速检查：

```bash
python scripts/run_sweep.py --episodes 10 --users expert noisy-0.6 --gammas 0 0.2 1.0 --out /tmp/smoke
```

说明：

- 官方代码依赖 CUDA、TensorFlow、tf-agents 和 d4rl。BlockPush 评测实际上用不到这些库，`sha_exp/compat.py` 会对它们做 stub，并把 SAC/扩散模型的加载改为 CPU 兼容，所以无需 docker，也无需 GPU。
- γ 的定义、加噪/去噪流程，以及预训练权重和超参数（50 步、sigmoid β 调度、β∈[1e-4, 0.26]），都与官方 `DiffusionAssistedActor` 和 `configs.json` 一致。
- 复现校验：在 γ=0.4 时，专家用户的正确目标率约为 0.5–0.6，与官方仓库 `table_analysis/summary/BP-*.csv` 中的 58/100 一致；γ≥0.5 时，正确与错误目标接近 50/50，这也和官方表格相符。

### 6. 目录结构

```
sha_exp/compat.py    # 让官方 diffusha 在 CPU 上可导入（stub 掉 TF/tf-agents/d4rl，修补 GPU 硬编码）
sha_exp/core.py      # 加载预训练 SAC 专家 + 扩散模型；assist()：部分前向扩散 + 条件去噪；run_episode()
sha_exp/users.py     # 模拟用户
scripts/run_sweep.py # 用户 × γ 网格评测
scripts/analyze.py   # 第一轮分析
scripts/bandit_sim.py# 后续选题：有限反馈下的 γ 选择
results/             # 逐 episode 数据、汇总表、图
```
