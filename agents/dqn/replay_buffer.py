"""Uniform replay buffer for DQN."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Tuple

import numpy as np
import torch


@dataclass
class Transition:
    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int = 50_000) -> None:
        self.capacity = int(capacity)
        self.buffer: Deque[Transition] = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append(
            Transition(
                obs=np.asarray(obs, dtype=np.float32),
                action=int(action),
                reward=float(reward),
                next_obs=np.asarray(next_obs, dtype=np.float32),
                done=bool(done),
            )
        )

    def sample(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        batch = [self.buffer[i] for i in idx]

        obs = torch.tensor(np.stack([t.obs for t in batch]), dtype=torch.float32, device=device)
        actions = torch.tensor([t.action for t in batch], dtype=torch.long, device=device)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=device)
        next_obs = torch.tensor(np.stack([t.next_obs for t in batch]), dtype=torch.float32, device=device)
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32, device=device)

        return obs, actions, rewards, next_obs, dones
