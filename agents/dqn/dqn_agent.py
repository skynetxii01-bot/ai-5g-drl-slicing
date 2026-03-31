import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from agents.dqn.dqn_network import DuelingDQN
from agents.dqn.replay_buffer import ReplayBuffer


class DQNAgent:
    def __init__(self, state_dim=15, action_dim=27, device="cpu"):
        self.device = torch.device(device)
        self.q = DuelingDQN(state_dim, action_dim).to(self.device)
        self.tgt = DuelingDQN(state_dim, action_dim).to(self.device)
        self.tgt.load_state_dict(self.q.state_dict())
        self.opt = optim.Adam(self.q.parameters(), lr=1e-3)
        self.buf = ReplayBuffer(50000)
        self.gamma, self.batch, self.tau = 0.99, 64, 0.005
        self.eps_start, self.eps_end, self.eps_decay_steps = 1.0, 0.01, 20000
        self.step = 0
        self.action_dim = action_dim

    def act(self, obs, eval_mode=False):
        eps = self.eps_end + (self.eps_start - self.eps_end) * max(0, 1 - self.step / self.eps_decay_steps)
        if (not eval_mode) and np.random.rand() < eps:
            return np.random.randint(self.action_dim)
        with torch.no_grad():
            t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            return int(self.q(t).argmax(-1).item())

    def learn(self):
        if len(self.buf) < self.batch:
            return None
        s, a, r, ns, d = self.buf.sample(self.batch)
        s = torch.tensor(s, dtype=torch.float32, device=self.device)
        a = torch.tensor(a, dtype=torch.long, device=self.device).unsqueeze(-1)
        r = torch.tensor(r, dtype=torch.float32, device=self.device).unsqueeze(-1)
        ns = torch.tensor(ns, dtype=torch.float32, device=self.device)
        d = torch.tensor(d, dtype=torch.float32, device=self.device).unsqueeze(-1)
        q = self.q(s).gather(1, a)
        na = self.q(ns).argmax(-1, keepdim=True)
        nq = self.tgt(ns).gather(1, na)
        y = r + self.gamma * (1 - d) * nq
        loss = F.mse_loss(q, y.detach())
        self.opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(self.q.parameters(), 10.0); self.opt.step()
        for tp, p in zip(self.tgt.parameters(), self.q.parameters()):
            tp.data.copy_(tp.data * (1 - self.tau) + p.data * self.tau)
        return float(loss.item())

    def update(self, s, a, r, ns, done):
        self.buf.push(s, a, r, ns, done)
        self.step += 1
        return self.learn()
