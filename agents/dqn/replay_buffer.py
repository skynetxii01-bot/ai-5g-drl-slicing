"""
agents/dqn/replay_buffer.py
============================
Experience replay buffer for DQN.
Stores (obs, action, reward, next_obs, done) tuples.
Uniformly samples random minibatches for training.
"""

import numpy as np
import torch


class ReplayBuffer:
    """
    Fixed-size circular replay buffer.

    Parameters
    ----------
    capacity : int
        Maximum number of transitions to store (oldest are overwritten).
    obs_dim : int
        Dimension of observation vector.
    device : torch.device
        Device to return tensors on.
    """

    def __init__(self, capacity=50_000, obs_dim=15, device=None):
        self.capacity  = capacity
        self.obs_dim   = obs_dim
        self.device    = device or torch.device('cpu')
        self._ptr      = 0      # write pointer
        self._size     = 0      # current number of stored transitions

        # Pre-allocate numpy arrays for efficiency
        self._obs      = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._actions  = np.zeros(capacity, dtype=np.int64)
        self._rewards  = np.zeros(capacity, dtype=np.float32)
        self._dones    = np.zeros(capacity, dtype=np.float32)

    def add(self, obs, action, reward, next_obs, done):
        """Store one transition."""
        self._obs[self._ptr]      = obs
        self._actions[self._ptr]  = action
        self._rewards[self._ptr]  = reward
        self._next_obs[self._ptr] = next_obs
        self._dones[self._ptr]    = float(done)

        self._ptr  = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size=64):
        """Sample a random minibatch. Returns torch tensors."""
        idx = np.random.randint(0, self._size, size=batch_size)

        return (
            torch.FloatTensor(self._obs[idx]).to(self.device),
            torch.LongTensor(self._actions[idx]).to(self.device),
            torch.FloatTensor(self._rewards[idx]).to(self.device),
            torch.FloatTensor(self._next_obs[idx]).to(self.device),
            torch.FloatTensor(self._dones[idx]).to(self.device),
        )

    def __len__(self):
        return self._size

    @property
    def ready(self):
        """True when buffer has enough samples to start training."""
        return self._size >= 1000
