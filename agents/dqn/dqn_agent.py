"""
agents/dqn/dqn_agent.py
========================
Dueling DQN agent with:
- ε-greedy exploration (1.0 → 0.01 over 20k steps)
- Soft target network updates (τ=0.005)
- Gradient clipping for stability
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from .dqn_network import DuelingDQN
from .replay_buffer import ReplayBuffer


class DQNAgent:
    """
    Dueling DQN agent for 5G NR slice PRB allocation.

    Parameters
    ----------
    obs_dim    : Observation dimension (15)
    n_actions  : Number of discrete actions (27)
    lr         : Learning rate (1e-3)
    gamma      : Discount factor (0.99)
    tau        : Soft update coefficient for target net (0.005)
    buffer_cap : Replay buffer capacity (50_000)
    batch_size : Training batch size (64)
    eps_start  : Initial exploration ε (1.0)
    eps_end    : Final exploration ε (0.01)
    eps_decay  : Steps to decay ε over (20_000)
    device     : torch device
    """

    def __init__(
        self,
        obs_dim=15, n_actions=27,
        lr=1e-3, gamma=0.99, tau=0.005,
        buffer_cap=50_000, batch_size=64,
        eps_start=1.0, eps_end=0.01, eps_decay=20_000,
        device=None,
    ):
        self.n_actions  = n_actions
        self.gamma      = gamma
        self.tau        = tau
        self.batch_size = batch_size
        self.eps        = eps_start
        self.eps_end    = eps_end
        self.eps_decay  = eps_decay
        self.total_steps = 0
        self.device     = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        # Online network (trained every step)
        self.online_net = DuelingDQN(obs_dim, n_actions).to(self.device)
        # Target network (updated softly — provides stable training targets)
        self.target_net = DuelingDQN(obs_dim, n_actions).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.buffer    = ReplayBuffer(buffer_cap, obs_dim, self.device)
        self.loss_fn   = nn.SmoothL1Loss()  # Huber loss — robust to outliers

        print(f"DQNAgent initialised on {self.device}. "
              f"Params: {sum(p.numel() for p in self.online_net.parameters()):,}")

    def select_action(self, obs):
        """
        ε-greedy action selection.
        With probability ε: random action (exploration)
        Otherwise: greedy action from Q-network (exploitation)
        """
        if np.random.random() < self.eps:
            return np.random.randint(self.n_actions)

        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.online_net(obs_t)
        return q_values.argmax(dim=1).item()

    def store(self, obs, action, reward, next_obs, done):
        """Store transition and decay epsilon."""
        self.buffer.add(obs, action, reward, next_obs, done)
        self.total_steps += 1

        # Linear epsilon decay
        self.eps = max(
            self.eps_end,
            1.0 - self.total_steps / self.eps_decay
        )

    def train_step(self):
        """
        One gradient update step.
        Returns loss value (float) or None if buffer not ready.
        """
        if not self.buffer.ready:
            return None

        obs, actions, rewards, next_obs, dones = self.buffer.sample(self.batch_size)

        # Current Q-values for taken actions
        q_current = self.online_net(obs).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values using Double DQN:
        # - online net selects action, target net evaluates it
        with torch.no_grad():
            next_actions = self.online_net(next_obs).argmax(1)
            q_next       = self.target_net(next_obs).gather(
                               1, next_actions.unsqueeze(1)).squeeze(1)
            q_target = rewards + self.gamma * q_next * (1 - dones)

        loss = self.loss_fn(q_current, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping prevents exploding gradients
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Soft update target network: θ_target = τ·θ_online + (1-τ)·θ_target
        for online_p, target_p in zip(
                self.online_net.parameters(), self.target_net.parameters()):
            target_p.data.copy_(
                self.tau * online_p.data + (1 - self.tau) * target_p.data)

        return loss.item()

    def save(self, path):
        """Save model weights."""
        torch.save({
            'online_net': self.online_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer':  self.optimizer.state_dict(),
            'total_steps': self.total_steps,
            'eps': self.eps,
        }, path)
        print(f"DQN model saved to {path}")

    def load(self, path):
        """Load model weights."""
        ckpt = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ckpt['online_net'])
        self.target_net.load_state_dict(ckpt['target_net'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.total_steps = ckpt.get('total_steps', 0)
        self.eps         = ckpt.get('eps', self.eps_end)
        print(f"DQN model loaded from {path}")
