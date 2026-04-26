"""
TrimmedTrafficEnv
=================
Gymnasium wrapper that reduces the full 156-dim observation and 12-TL
action space of BakuSUMOEnv to the active topology of each scenario.

Motivation (thesis §3.2):
  Training on the full space introduces gradient noise from zero-padded,
  unused slots.  Trimming to the active TL count makes the learning
  problem tractable for a standard MLP policy and reduces convergence
  time by an order of magnitude (thesis §4.4 ablation).
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from envs.baku_sumo_env import BakuSUMOEnv


class TrimmedTrafficEnv(gym.Wrapper):
    """Slice observation and action to the active n_tls TLs."""

    def __init__(self, env: BakuSUMOEnv):
        super().__init__(env)
        n_tls    = env.n_tls
        obs_trim = n_tls * env.OBS_PER_TL

        # Trimmed observation space
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(obs_trim,),
            dtype=np.float32,
        )

        # Trimmed action space — active TLs only
        # n_phases_per_tl may still be the placeholder [6] × MAX_TLS until
        # the first reset triggers _discover_topology; but since we pass
        # n_phases_per_tl from scenario configs it is already correct.
        nvec = env.action_space.nvec[:n_tls]
        self.action_space = spaces.MultiDiscrete(nvec)

        self._n_tls     = n_tls
        self._obs_trim  = obs_trim
        self._max_tls   = env.MAX_TLS

    # ── Gymnasium API ────────────────────────────────────────────────────

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._trim_obs(obs), info

    def step(self, action):
        full_action    = np.zeros(self._max_tls, dtype=np.int64)
        full_action[:self._n_tls] = action
        obs, reward, done, trunc, info = self.env.step(full_action)
        return self._trim_obs(obs), reward, done, trunc, info

    # ── Delegate metrics helper ──────────────────────────────────────────

    def get_metrics(self) -> dict:
        return self.env.get_metrics()

    # ── Internal ─────────────────────────────────────────────────────────

    def _trim_obs(self, obs: np.ndarray) -> np.ndarray:
        return obs[: self._obs_trim].copy()
