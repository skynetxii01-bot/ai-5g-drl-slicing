import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    def __init__(self, state_dim=15, action_dim=27):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 256), nn.LayerNorm(256), nn.ReLU(),
        )
        self.actor = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, action_dim))
        self.critic = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, x):
        h = self.shared(x)
        return self.actor(h), self.critic(h)
