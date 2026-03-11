"""Rollout storage and GAE for PPO."""

import numpy as np


class RolloutBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.logprobs = []

    def add(self, obs, action, reward, done, value, logprob):
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.logprobs.append(logprob)

    def compute_gae(self, last_value, gamma=0.99, lam=0.95):
        values = self.values + [last_value]
        gae = 0.0
        returns = []
        for t in reversed(range(len(self.rewards))):
            delta = self.rewards[t] + gamma * values[t + 1] * (1.0 - self.dones[t]) - values[t]
            gae = delta + gamma * lam * (1.0 - self.dones[t]) * gae
            returns.insert(0, gae + values[t])
        adv = np.array(returns) - np.array(self.values)
        return np.array(returns, dtype=np.float32), adv.astype(np.float32)
