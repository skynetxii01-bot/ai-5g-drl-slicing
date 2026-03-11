"""
agents/r2d2/r2d2_network.py
============================
R2D2 network: FC → LSTM → Dueling heads
LSTM hidden state persists within episode → anticipates burst traffic patterns
"""
import torch
import torch.nn as nn


class R2D2Network(nn.Module):
    def __init__(self, obs_dim=15, n_actions=27, hidden=256):
        super().__init__()
        self.hidden = hidden

        self.feature = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128),    nn.ReLU(),
        )
        self.lstm = nn.LSTM(128, hidden, batch_first=True)

        self.value_head = nn.Sequential(
            nn.Linear(hidden, 128), nn.ReLU(), nn.Linear(128, 1))
        self.adv_head   = nn.Sequential(
            nn.Linear(hidden, 128), nn.ReLU(), nn.Linear(128, n_actions))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, hidden_state=None):
        """
        x: (batch, seq_len, obs_dim) or (batch, obs_dim) for single step
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # add seq dim

        feat = self.feature(x.reshape(-1, x.size(-1)))
        feat = feat.reshape(x.size(0), x.size(1), -1)

        lstm_out, hidden_state = self.lstm(feat, hidden_state)

        value = self.value_head(lstm_out)
        adv   = self.adv_head(lstm_out)
        q     = value + adv - adv.mean(dim=-1, keepdim=True)
        return q, hidden_state

    def init_hidden(self, batch_size=1, device='cpu'):
        return (torch.zeros(1, batch_size, self.hidden).to(device),
                torch.zeros(1, batch_size, self.hidden).to(device))
