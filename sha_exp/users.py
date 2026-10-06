"""Simulated users (surrogate pilots) for BlockPushMultimodal-v1.

Every user is the SAC expert (which pushes the block to its intended goal, 'target')
corrupted by a noise model. `noisy` and `laggy` reproduce the paper's surrogate
pilots exactly (diffusha/actor/base.py); `gauss` is an extra continuous-noise user.
Each user owns its RNG so episodes are reproducible and paired across assistance levels.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class UserSpec:
    kind: str  # 'expert' | 'noisy' | 'laggy' | 'gauss'
    level: float = 0.0  # eps for noisy, repeat prob for laggy, sigma for gauss

    @property
    def name(self) -> str:
        return "expert" if self.kind == "expert" else f"{self.kind}-{self.level:g}"

    @staticmethod
    def parse(name: str) -> "UserSpec":
        if name == "expert":
            return UserSpec("expert")
        kind, level = name.rsplit("-", 1)
        return UserSpec(kind, float(level))


# Default population: paper's two noise types at several severities + Gaussian jitter.
DEFAULT_USERS = [
    "expert",
    "noisy-0.2", "noisy-0.4", "noisy-0.6", "noisy-0.8",
    "laggy-0.3", "laggy-0.5", "laggy-0.7", "laggy-0.85",
    "gauss-0.5", "gauss-1",
]


class SimulatedUser:
    def __init__(self, spec: UserSpec, expert_agent, act_space, seed: int = 0):
        self.spec = spec
        self.expert = expert_agent
        self.low, self.high = act_space.low, act_space.high
        self.rng = np.random.default_rng(seed)
        self.prev_action = None

    def reset(self):
        self.prev_action = self._random_action()

    def _random_action(self):
        return self.rng.uniform(self.low, self.high).astype(np.float32)

    def _expert_action(self, obs):
        return np.asarray(self.expert.act(obs), dtype=np.float32)

    def act(self, obs) -> np.ndarray:
        kind, level = self.spec.kind, self.spec.level
        if kind == "expert":
            action = self._expert_action(obs)
        elif kind == "noisy":
            # With prob eps, output a uniformly random action (paper's NoisyActor).
            action = self._random_action() if self.rng.random() < level else self._expert_action(obs)
        elif kind == "laggy":
            # With prob p, repeat the previous action (paper's LaggyActor).
            action = self.prev_action if self.rng.random() < level else self._expert_action(obs)
        elif kind == "gauss":
            action = self._expert_action(obs) + level * self.rng.standard_normal(self.low.shape).astype(np.float32)
            action = np.clip(action, self.low, self.high)
        else:
            raise ValueError(f"unknown user kind: {kind}")
        self.prev_action = action
        return action
