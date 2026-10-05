"""Loading the official pretrained models and running assisted episodes.

Assistance follows the paper (Alg. 1, `DiffusionAssistedActor` in the official repo):
the user action a_h is forward-diffused for k = floor((T-1) * gamma) steps, then
denoised for k steps with the observation clamped as a hard condition.
gamma = 0 means no assistance (action passed through unchanged).
"""
from __future__ import annotations

from . import compat  # noqa: F401  (must come before diffusha imports)

import numpy as np
import torch

ENV_NAME = "BlockPushMultimodal-v1"
USER_GOAL = "target"
EPISODE_LIMIT = 100  # registered max_episode_steps of BlockPushMultimodal-v1

# Hyperparameters of the released checkpoint (diffusha/diffusion/evaluation/configs.json)
NUM_DIFFUSION_STEPS = 50
BETA_SCHEDULE, BETA_MIN, BETA_MAX = "sigmoid", 1e-4, 0.26
DDPM_STEP = 29999


def make_env(seed: int = 0):
    from diffusha.data_collection.env import make_env as _make_env

    return _make_env(ENV_NAME, test=True, seed=seed, user_goal=USER_GOAL)


def load_expert():
    from diffusha.data_collection.utils.agent import get_agent

    agent = get_agent(make_env(), policy_output_scale=1.0, batch_size=256, replay_start_size=10_000)
    agent.load(str(compat.DATA_DIR / "experts" / "blockpush"))
    agent.training = False
    return agent


def load_diffusion():
    from diffusha.diffusion.ddpm import DiffusionCore, DiffusionModel

    env = make_env()
    obs_size = env.observation_space.low.size
    act_size = env.action_space.low.size
    diffusion = DiffusionModel(
        diffusion_core=DiffusionCore(),
        num_diffusion_steps=NUM_DIFFUSION_STEPS,
        input_size=obs_size + act_size,
        beta_schedule=BETA_SCHEDULE,
        beta_min=BETA_MIN,
        beta_max=BETA_MAX,
        cond_dim=obs_size,
    )
    ckpt = compat.DATA_DIR / "ddpm" / ENV_NAME.lower() / f"step_{DDPM_STEP:08d}.pt"
    state = torch.load(ckpt, map_location=diffusion.device)
    diffusion.model.load_state_dict(state["model"])
    diffusion.model.eval()
    return diffusion


def gamma_to_k(diffusion, gamma: float) -> int:
    assert 0.0 <= gamma <= 1.0
    return int((diffusion.num_diffusion_steps - 1) * gamma)


@torch.no_grad()
def assist(diffusion, obs: np.ndarray, user_act: np.ndarray, gamma: float) -> np.ndarray:
    """Partial forward diffusion of the user action + conditional reverse diffusion."""
    k = gamma_to_k(diffusion, gamma)
    if k == 0:
        return user_act
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=diffusion.device)
    x0 = torch.cat([obs_t, torch.as_tensor(user_act, dtype=torch.float32, device=diffusion.device)])[None]
    x, _ = diffusion.diffuse(x0, torch.as_tensor([k]))
    x[:, : obs_t.numel()] = obs_t
    for i in reversed(range(k)):
        x = diffusion.p_sample(x, i)
        x[:, : obs_t.numel()] = obs_t
    return x[0, obs_t.numel():].cpu().numpy()


def seed_episode(env, user, episode_seed: int):
    """Common random numbers: the same episode_seed gives the same initial scene, the
    same user-noise stream and the same diffusion noise, whatever gamma is used.
    This makes comparisons across assistance levels paired (much lower variance)."""
    env.seed(episode_seed)
    user.rng = np.random.default_rng(episode_seed)
    torch.manual_seed(episode_seed)


def run_episode(env, user, diffusion, gamma: float) -> dict:
    """Roll out one episode. Returns outcome + diagnostics.

    outcome: 'correct' (block reached the user's goal), 'wrong' (reached the other goal),
             'timeout' (no goal within EPISODE_LIMIT steps).
    """
    obs = env.reset()
    user.reset()
    done, steps, ret, diffs = False, 0, 0.0, []
    info = {}
    while not done:
        a_user = user.act(obs)
        a_exec = assist(diffusion, obs, a_user, gamma)
        diffs.append(float(np.linalg.norm(a_exec - a_user)))
        obs, rew, done, info = env.step(a_exec)
        ret += rew
        steps += 1
    if info.get("finished"):
        outcome = "correct"
    elif info.get("state", "NotDone") != "NotDone":
        outcome = "wrong"
    else:
        outcome = "timeout"
    return {
        "outcome": outcome,
        "success": int(outcome == "correct"),
        "ep_len": steps,
        "return": ret,
        "action_diff": float(np.mean(diffs)),
    }
