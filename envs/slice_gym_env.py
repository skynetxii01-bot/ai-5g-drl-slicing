"""Gym 0.21-compatible wrapper for the NS-3 slice OpenGym environment."""

from __future__ import annotations

import random
from typing import Dict, Tuple

import gym
import numpy as np
from gym import spaces

from ns3gym import ns3env


class SliceGymEnv(gym.Env):
    """Thin adapter around ns3-gym with helper baseline policies."""

    metadata = {"render.modes": []}

    def __init__(self, port: int = 5555, seed: int = 42, step_time: float = 0.1, start_sim: bool = False):
        super().__init__()
        self.port = port
        self.seed_value = seed
        self.step_time = step_time
        self.action_space = spaces.Discrete(27)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(15,), dtype=np.float32)
        self._rng = np.random.default_rng(seed)
        self._ns3_env = ns3env.Ns3Env(port=port, stepTime=step_time, startSim=start_sim, seed=seed)

    def seed(self, seed: int | None = None):
        if seed is not None:
            self.seed_value = seed
        random.seed(self.seed_value)
        np.random.seed(self.seed_value)
        self._rng = np.random.default_rng(self.seed_value)
        return [self.seed_value]

    def _decode_obs(self, obs) -> Dict[str, np.ndarray]:
        arr = np.asarray(obs, dtype=np.float32).reshape(15)
        return {
            "raw": arr,
            "prb_fraction": arr[0:3],
            "throughput_norm": arr[3:6],
            "latency_norm": arr[6:9],
            "queue_occupancy": arr[9:12],
            "ue_fraction": arr[12:15],
        }

    def reset(self):
        obs = self._ns3_env.reset()
        return self._decode_obs(obs)["raw"]

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        obs, reward, done, info = self._ns3_env.step(int(action))
        decoded = self._decode_obs(obs)
        info = dict(info or {})
        info.update(decoded)
        return decoded["raw"], float(reward), bool(done), info

    def close(self):
        self._ns3_env.close()

    def random_policy(self, _obs: np.ndarray) -> int:
        return int(self.action_space.sample())

    def round_robin_policy(self, obs: np.ndarray) -> int:
        # Cycle through all possible 27 actions based on step-local signal.
        index = int(np.argmax(obs[0:3]))
        return (index * 9 + int(self._rng.integers(0, 9))) % 27

    def proportional_fair_policy(self, obs: np.ndarray) -> int:
        thr = obs[3:6]
        deficit = 1.0 - thr
        target = int(np.argmax(deficit))
        # Map to actions where target slice gets +1 and others 0/-1 when possible.
        action_map = {
            0: 22,  # (+1,0,0)
            1: 16,  # (0,+1,0)
            2: 14,  # (0,0,+1)
        }
        return action_map[target]
