"""Actor-critic network for PPO with shared backbone."""

import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int = 15, act_dim: int = 27):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
        )
        self.actor = nn.Linear(256, act_dim)
        self.critic = nn.Linear(256, 1)

    def forward(self, obs: torch.Tensor):
        h = self.backbone(obs)
        return self.actor(h), self.critic(h)

    def get_action_value(self, obs: torch.Tensor):
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        logprob = dist.log_prob(action)
        return action, logprob, value.squeeze(-1)
