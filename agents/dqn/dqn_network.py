import torch
import torch.nn as nn


class DuelingDQN(nn.Module):
    def __init__(self, state_dim=15, action_dim=27):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
        )
        self.value = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1))
        self.adv = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, action_dim))

    def forward(self, x):
        z = self.backbone(x)
        v = self.value(z)
        a = self.adv(z)
        return v + a - a.mean(dim=-1, keepdim=True)
