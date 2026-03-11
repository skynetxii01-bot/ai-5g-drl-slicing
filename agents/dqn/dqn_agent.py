"""DQN agent with soft target updates and epsilon decay."""

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from agents.dqn.dqn_network import DuelingDQN
from agents.dqn.replay_buffer import ReplayBuffer


class DQNAgent:
    def __init__(self, obs_dim: int = 15, act_dim: int = 27, device: str = "cpu"):
        self.device = torch.device(device)
        self.act_dim = act_dim
        self.gamma = 0.99
        self.tau = 0.005
        self.batch_size = 128

        self.online = DuelingDQN(obs_dim, act_dim).to(self.device)
        self.target = DuelingDQN(obs_dim, act_dim).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = optim.Adam(self.online.parameters(), lr=1e-3)

        self.buffer = ReplayBuffer(capacity=50000)
        self.total_steps = 0
        self.eps_start = 1.0
        self.eps_end = 0.01
        self.eps_decay_steps = 20000

    def epsilon(self) -> float:
        frac = min(1.0, self.total_steps / self.eps_decay_steps)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def select_action(self, obs: np.ndarray, eval_mode: bool = False) -> int:
        self.total_steps += 0 if eval_mode else 1
        if (not eval_mode) and np.random.rand() < self.epsilon():
            return np.random.randint(0, self.act_dim)
        with torch.no_grad():
            q = self.online(torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0))
            return int(torch.argmax(q, dim=1).item())

    def store(self, s, a, r, ns, d):
        self.buffer.push(s, a, r, ns, d)

    def update(self):
        if len(self.buffer) < self.batch_size:
            return None
        s, a, r, ns, d = self.buffer.sample(self.batch_size)
        s = torch.tensor(s, dtype=torch.float32, device=self.device)
        a = torch.tensor(a, dtype=torch.long, device=self.device).unsqueeze(1)
        r = torch.tensor(r, dtype=torch.float32, device=self.device).unsqueeze(1)
        ns = torch.tensor(ns, dtype=torch.float32, device=self.device)
        d = torch.tensor(d, dtype=torch.float32, device=self.device).unsqueeze(1)

        q = self.online(s).gather(1, a)
        with torch.no_grad():
            next_q = self.target(ns).max(dim=1, keepdim=True)[0]
            target_q = r + self.gamma * (1.0 - d) * next_q

        loss = F.mse_loss(q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        for tp, op in zip(self.target.parameters(), self.online.parameters()):
            tp.data.copy_(tp.data * (1.0 - self.tau) + op.data * self.tau)

        return float(loss.item())
