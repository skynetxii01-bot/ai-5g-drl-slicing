"""
agents/ppo/actor_critic.py
===========================
Shared backbone Actor-Critic network for PPO.
"""
import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, obs_dim=15, n_actions=27, hidden=256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(),
        )
        self.actor  = nn.Linear(hidden, n_actions)  # policy logits
        self.critic = nn.Linear(hidden, 1)           # value function V(s)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.0)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)

    def forward(self, x):
        feat   = self.backbone(x)
        logits = self.actor(feat)
        value  = self.critic(feat).squeeze(-1)
        return logits, value

    def get_action(self, obs):
        logits, value = self(obs)
        dist   = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    def evaluate(self, obs, actions):
        logits, value = self(obs)
        dist   = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value
