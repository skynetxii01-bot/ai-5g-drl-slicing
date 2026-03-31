import itertools
from typing import Dict, Tuple

import gym
import numpy as np
from gym import spaces

SLICE_NAMES = ["eMBB", "URLLC", "mMTC"]
ACTION_MAP = list(itertools.product([-1, 0, 1], repeat=3))


class SliceGymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, port: int = 5555, seed: int = 42):
        super().__init__()
        self.port = port
        self._rng = np.random.default_rng(seed)
        self.total_prbs = 25
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(15,), dtype=np.float32)
        self.action_space = spaces.Discrete(27)
        self._obs = np.zeros(15, dtype=np.float32)
        self._step = 0
        self._max_steps = 1000

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._step = 0
        self._obs = np.array([10, 8, 7] + [0] * 12, dtype=np.float32)
        self._obs[:3] /= self.total_prbs
        return self._obs.copy(), {"port": self.port}

    def step(self, action: int):
        action = int(action)
        deltas = np.array(ACTION_MAP[action], dtype=np.float32)
        prbs = np.maximum(1, np.round(self._obs[:3] * self.total_prbs + deltas)).astype(np.float32)
        diff = self.total_prbs - int(prbs.sum())
        while diff != 0:
            idx = int(np.argmin(prbs) if diff > 0 else np.argmax(prbs))
            if diff < 0 and prbs[idx] <= 1:
                break
            prbs[idx] += 1 if diff > 0 else -1
            diff += -1 if diff > 0 else 1
        self._obs[:3] = prbs / self.total_prbs
        thr = np.clip(self._obs[:3] * np.array([1.0, 0.8, 0.6]) + self._rng.normal(0, 0.02, 3), 0, 1)
        lat = np.clip(1.0 - self._obs[:3] + self._rng.normal(0, 0.02, 3), 0, 1)
        q = np.clip(1.0 - self._obs[:3], 0, 1)
        ue = np.array([10, 5, 20], dtype=np.float32) / 35.0
        self._obs[3:6], self._obs[6:9], self._obs[9:12], self._obs[12:15] = thr, lat, q, ue
        min_thr = np.array([0.1, 0.1, 0.05])
        max_lat = np.array([1.0, 0.1, 1.0])
        sla_viol = ((thr < min_thr) | (lat > max_lat)).sum()
        jain = (thr.sum() ** 2) / (3 * (thr ** 2).sum() + 1e-9)
        reward = 0.5 * thr.mean() + 0.3 * np.mean((thr >= min_thr) * (1 - lat)) + 0.2 * jain - 2.0 * sla_viol
        self._step += 1
        done = self._step >= self._max_steps
        return self._obs.copy(), float(reward), done, False, self._decode_obs(self._obs)

    def _decode_obs(self, obs: np.ndarray) -> Dict[str, Dict[str, float]]:
        return {
            "prb_frac": dict(zip(SLICE_NAMES, obs[0:3].tolist())),
            "throughput": dict(zip(SLICE_NAMES, obs[3:6].tolist())),
            "latency": dict(zip(SLICE_NAMES, obs[6:9].tolist())),
            "queue_occ": dict(zip(SLICE_NAMES, obs[9:12].tolist())),
            "ue_count": dict(zip(SLICE_NAMES, obs[12:15].tolist())),
        }
