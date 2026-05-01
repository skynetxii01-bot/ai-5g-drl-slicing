"""Gym 0.26.2 wrapper for the NS-3 5G slice RL environment."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gym
import numpy as np
from gym import spaces
from ns3gym import ns3env

SLICE_NAMES = ("eMBB", "URLLC", "mMTC")
OBS_SIZE = 15
ACTION_SIZE = 27


class SliceGymEnv(gym.Env):
    """Gym 0.26-compatible wrapper over ns3-gym for 3-slice PRB control."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        port: int = 5555,
        sim_seed: int = 42,
        debug: bool = False,
        start_sim: bool = False,
        sim_args: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()

        self.port = int(port)
        self.sim_seed = int(sim_seed)
        self.debug = bool(debug)
        self.start_sim = bool(start_sim)
        self.sim_args = sim_args or {}

        self.observation_space = spaces.Box(
            low=np.float32(0.0),
            high=np.float32(1.0),
            shape=(OBS_SIZE,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(ACTION_SIZE)

        self._ns3_env = ns3env.Ns3Env(
            port=self.port,
            simSeed=self.sim_seed,
            simArgs=self.sim_args,
            startSim=self.start_sim,
            debug=self.debug,
        )

    def _validate_obs(self, obs: np.ndarray) -> np.ndarray:
        arr = np.asarray(obs, dtype=np.float32).reshape(-1)
        if arr.shape[0] != OBS_SIZE:
            raise ValueError(
                f"Expected EXACT {OBS_SIZE}-dim observation from NS-3, got {arr.shape[0]}"
            )
        return arr

    def _decode_obs(self, obs: np.ndarray) -> Dict[str, Dict[str, float]]:
        arr = self._validate_obs(obs)
        return {
            "prb_frac": dict(zip(SLICE_NAMES, arr[0:3].tolist())),
            "throughput": dict(zip(SLICE_NAMES, arr[3:6].tolist())),
            "latency": dict(zip(SLICE_NAMES, arr[6:9].tolist())),
            "queue_occ": dict(zip(SLICE_NAMES, arr[9:12].tolist())),
            "ue_count": dict(zip(SLICE_NAMES, arr[12:15].tolist())),
        }

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        del options
        if seed is not None:
            self.sim_seed = int(seed)

        raw = self._ns3_env.reset()

        # ns3-gym wrappers differ by version: old returns obs, newer may return (obs, info)
        if isinstance(raw, tuple) and len(raw) == 2:
            obs_raw, info = raw
            info = dict(info)
        else:
            obs_raw, info = raw, {}

        obs = self._validate_obs(obs_raw)
        info["decoded_obs"] = self._decode_obs(obs)
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        act = int(action)
        if act < 0 or act >= ACTION_SIZE:
            raise ValueError(f"Action must be in [0, {ACTION_SIZE - 1}], got {act}")

        raw = self._ns3_env.step(act)

        # Handle both old (obs, reward, done, info) and 0.26-like 5-value returns.
        if isinstance(raw, tuple) and len(raw) == 5:
            obs_raw, reward, done, truncated, info = raw
        elif isinstance(raw, tuple) and len(raw) == 4:
            obs_raw, reward, done, info = raw
            truncated = False
        else:
            raise RuntimeError("Unexpected step() return format from ns3-gym backend")

        obs = self._validate_obs(obs_raw)
        info = dict(info)
        info["decoded_obs"] = self._decode_obs(obs)

        return obs, float(reward), bool(done), bool(truncated), info

    def render(self) -> None:
        return None

    def close(self) -> None:
        if self._ns3_env is not None:
            self._ns3_env.close()


__all__ = ["SliceGymEnv", "SLICE_NAMES", "OBS_SIZE", "ACTION_SIZE"]
