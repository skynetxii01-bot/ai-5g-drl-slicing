"""
agents/ppo/rollout_buffer.py
"""
import numpy as np
import torch


class RolloutBuffer:
    def __init__(self, n_steps=2048, obs_dim=15, gamma=0.99, lam=0.95, device=None):
        self.n_steps = n_steps
        self.gamma   = gamma
        self.lam     = lam
        self.device  = device or torch.device('cpu')
        self.reset()

    def reset(self):
        self.obs      = []
        self.actions  = []
        self.rewards  = []
        self.values   = []
        self.log_probs= []
        self.dones    = []

    def add(self, obs, action, reward, value, log_prob, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def compute_returns(self, last_value):
        """Compute GAE advantages and returns."""
        advantages = np.zeros(len(self.rewards), dtype=np.float32)
        gae = 0
        values = np.array([v.item() for v in self.values] + [last_value])

        for t in reversed(range(len(self.rewards))):
            delta = self.rewards[t] + self.gamma * values[t+1] * (1 - self.dones[t]) - values[t]
            gae   = delta + self.gamma * self.lam * (1 - self.dones[t]) * gae
            advantages[t] = gae

        returns = advantages + np.array([v.item() for v in self.values])
        return (torch.FloatTensor(advantages).to(self.device),
                torch.FloatTensor(returns).to(self.device))

    def get(self):
        return (
            torch.FloatTensor(np.array(self.obs)).to(self.device),
            torch.LongTensor(self.actions).to(self.device),
            torch.FloatTensor(self.log_probs).to(self.device),
        )
