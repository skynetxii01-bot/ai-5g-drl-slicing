"""Simple Prioritized Experience Replay for sequence samples."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SequenceTransition:
    obs: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    done: np.ndarray
    next_obs: np.ndarray


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int = 100_000, alpha: float = 0.6) -> None:
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.data = []
        self.priorities = np.zeros(self.capacity, dtype=np.float32)
        self.pos = 0

    def __len__(self):
        return len(self.data)

    def add(self, item: SequenceTransition, priority: float = 1.0):
        p = max(1e-6, float(priority))
        if len(self.data) < self.capacity:
            self.data.append(item)
        else:
            self.data[self.pos] = item
        self.priorities[self.pos] = p
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = 0.4):
        n = len(self.data)
        prios = self.priorities[:n] ** self.alpha
        probs = prios / prios.sum()
        idx = np.random.choice(n, batch_size, p=probs)
        samples = [self.data[i] for i in idx]
        weights = (n * probs[idx]) ** (-beta)
        weights /= weights.max()
        return idx, samples, weights.astype(np.float32)

    def update_priorities(self, indices, priorities):
        for i, p in zip(indices, priorities):
            self.priorities[int(i)] = max(1e-6, float(p))
