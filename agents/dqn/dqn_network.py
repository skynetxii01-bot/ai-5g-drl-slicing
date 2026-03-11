"""
agents/dqn/dqn_network.py
==========================
Dueling DQN network architecture.

Architecture:
  Input(15) → FC(256)+LayerNorm+ReLU → FC(256)+LayerNorm+ReLU → FC(128)+ReLU
  → split into:
    Value stream:     FC(128) → FC(1)     → V(s)
    Advantage stream: FC(128) → FC(27)    → A(s,a)
  Q(s,a) = V(s) + A(s,a) - mean(A(s,a))

Dueling architecture helps separate "how good is this state" from
"which action is best in this state" — important for slicing where
many actions have similar effects.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingDQN(nn.Module):
    """
    Dueling Deep Q-Network for 5G NR slice resource allocation.

    Input:  15-dim observation vector
    Output: 27-dim Q-value vector (one per discrete action)
    """

    def __init__(self, obs_dim=15, n_actions=27, hidden=256):
        super().__init__()

        # Shared feature extraction backbone
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, 128),
            nn.ReLU(),
        )

        # Value stream: estimates V(s) — scalar
        self.value_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        # Advantage stream: estimates A(s,a) for each action
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

        # Weight initialisation (helps convergence)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        """
        Forward pass.
        x: tensor of shape (batch, 15)
        returns: Q-values of shape (batch, 27)
        """
        features  = self.backbone(x)
        value     = self.value_stream(features)        # (batch, 1)
        advantage = self.advantage_stream(features)    # (batch, 27)

        # Combine: Q = V + A - mean(A)
        # Subtracting mean makes Q identifiable (removes scale ambiguity)
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values
