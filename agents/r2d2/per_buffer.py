"""
agents/r2d2/per_buffer.py
==========================
Prioritised Experience Replay buffer for R2D2.
Sequences of length seq_len are stored and sampled by priority.
"""
import numpy as np
import torch


class PERBuffer:
    def __init__(self, capacity=10_000, seq_len=40, obs_dim=15,
                 alpha=0.6, beta=0.4, device=None):
        self.capacity = capacity
        self.seq_len  = seq_len
        self.alpha    = alpha
        self.beta     = beta
        self.device   = device or torch.device('cpu')

        self._obs      = np.zeros((capacity, seq_len, obs_dim), dtype=np.float32)
        self._actions  = np.zeros((capacity, seq_len), dtype=np.int64)
        self._rewards  = np.zeros((capacity, seq_len), dtype=np.float32)
        self._dones    = np.zeros((capacity, seq_len), dtype=np.float32)
        self._priorities = np.ones(capacity, dtype=np.float32)
        self._ptr = 0
        self._size = 0

    def add(self, obs_seq, act_seq, rew_seq, done_seq, priority=1.0):
        self._obs[self._ptr]      = obs_seq
        self._actions[self._ptr]  = act_seq
        self._rewards[self._ptr]  = rew_seq
        self._dones[self._ptr]    = done_seq
        self._priorities[self._ptr] = max(priority, 1e-6) ** self.alpha
        self._ptr  = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size=32):
        probs = self._priorities[:self._size]
        probs = probs / probs.sum()
        idx   = np.random.choice(self._size, batch_size, p=probs, replace=False)

        weights = (self._size * probs[idx]) ** (-self.beta)
        weights /= weights.max()

        return (
            torch.FloatTensor(self._obs[idx]).to(self.device),
            torch.LongTensor(self._actions[idx]).to(self.device),
            torch.FloatTensor(self._rewards[idx]).to(self.device),
            torch.FloatTensor(self._dones[idx]).to(self.device),
            torch.FloatTensor(weights).to(self.device),
            idx,
        )

    def update_priorities(self, idx, priorities):
        self._priorities[idx] = (priorities + 1e-6) ** self.alpha

    def __len__(self):
        return self._size
