"""Simple recurrent dueling Q-network for R2D2-style learning."""

import torch
import torch.nn as nn


class R2D2Network(nn.Module):
    def __init__(self, obs_dim: int = 15, act_dim: int = 27, hidden_dim: int = 256):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(obs_dim, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        self.lstm = nn.LSTM(128, hidden_dim, batch_first=True)
        self.v_head = nn.Sequential(nn.Linear(hidden_dim, 128), nn.ReLU(), nn.Linear(128, 1))
        self.a_head = nn.Sequential(nn.Linear(hidden_dim, 128), nn.ReLU(), nn.Linear(128, act_dim))

    def forward(self, x, hidden=None):
        x = self.fc(x)
        y, hidden = self.lstm(x, hidden)
        v = self.v_head(y)
        a = self.a_head(y)
        q = v + a - a.mean(dim=-1, keepdim=True)
        return q, hidden
