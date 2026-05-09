import torch
import torch.nn as nn


class R2D2Net(nn.Module):
    """Dueling LSTM network for R2D2 — 18-dim obs, 27 actions."""

    def __init__(self, state_dim: int = 18, action_dim: int = 27) -> None:
        super().__init__()
        self.enc  = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128),       nn.ReLU(),
        )
        self.lstm = nn.LSTM(128, 256, batch_first=True)
        self.val  = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
        self.adv  = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, action_dim))

    def forward(self, x: torch.Tensor, hc=None):
        z    = self.enc(x)
        y, hc = self.lstm(z, hc)
        v    = self.val(y)
        a    = self.adv(y)
        q    = v + a - a.mean(dim=-1, keepdim=True)
        return q, hc
