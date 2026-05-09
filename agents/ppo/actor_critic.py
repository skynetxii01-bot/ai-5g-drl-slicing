import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    """Shared-trunk actor-critic for 18-dim observation and 27 actions."""

    def __init__(self, state_dim: int = 18, action_dim: int = 27) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 256),       nn.LayerNorm(256), nn.ReLU(),
        )
        self.actor  = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, action_dim))
        self.critic = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, x: torch.Tensor):
        h = self.shared(x)
        return self.actor(h), self.critic(h)
