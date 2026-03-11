"""Prioritized replay for sequence-based recurrent updates."""

import numpy as np


class PrioritizedSequenceReplay:
    def __init__(self, capacity=50000, alpha=0.6, beta_start=0.4, beta_end=1.0):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.storage = []
        self.priorities = []

    def __len__(self):
        return len(self.storage)

    def add(self, transition, priority=1.0):
        if len(self.storage) >= self.capacity:
            self.storage.pop(0)
            self.priorities.pop(0)
        self.storage.append(transition)
        self.priorities.append(float(priority))

    def sample(self, batch_size, step, total_steps):
        probs = np.array(self.priorities, dtype=np.float32) ** self.alpha
        probs /= probs.sum()
        idx = np.random.choice(len(self.storage), batch_size, p=probs)
        beta = self.beta_start + (self.beta_end - self.beta_start) * min(1.0, step / max(1, total_steps))
        weights = (len(self.storage) * probs[idx]) ** (-beta)
        weights /= weights.max()
        batch = [self.storage[i] for i in idx]
        return batch, idx, weights.astype(np.float32)

    def update_priorities(self, idxs, priorities):
        for i, p in zip(idxs, priorities):
            self.priorities[i] = float(max(1e-6, p))
