"""Rollout buffer for PPO — stores one episode's worth of transitions."""

from __future__ import annotations

import numpy as np


class RolloutBuffer:
    def __init__(self) -> None:
        self.clear()

    def __len__(self) -> int:
        return len(self._obs)

    def add(
        self,
        obs,
        action:   int,
        log_prob: float,
        reward:   float,
        done:     bool,
        value:    float,
    ) -> None:
        """Store one transition.
        Order: obs, action, log_prob, reward, done, value.
        Must match the unpacking order in PpoAgent.update().
        """
        self._obs.append(np.asarray(obs, dtype=np.float32))
        self._actions.append(int(action))
        self._log_probs.append(float(log_prob))
        self._rewards.append(float(reward))
        self._dones.append(float(done))
        self._values.append(float(value))

    def get(self):
        """Return arrays ready for update().
        Returns: obs (np.ndarray), actions, log_probs, rewards, dones, values (lists).
        """
        return (
            np.stack(self._obs),
            self._actions,
            self._log_probs,
            self._rewards,
            self._dones,
            self._values,
        )

    def clear(self) -> None:
        self._obs      = []
        self._actions  = []
        self._log_probs = []
        self._rewards  = []
        self._dones    = []
        self._values   = []
