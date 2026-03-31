import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from agents.r2d2.per_buffer import PrioritizedReplay
from agents.r2d2.r2d2_network import R2D2Net


class R2D2Agent:
    def __init__(self, state_dim=15, action_dim=27, device="cpu"):
        self.device = torch.device(device)
        self.q = R2D2Net(state_dim, action_dim).to(self.device)
        self.tgt = R2D2Net(state_dim, action_dim).to(self.device)
        self.tgt.load_state_dict(self.q.state_dict())
        self.opt = optim.Adam(self.q.parameters(), lr=1e-4)
        self.buf = PrioritizedReplay(100000, alpha=0.6)
        self.gamma, self.batch, self.beta = 0.99, 32, 0.4
        self.action_dim = action_dim

    def act(self, obs, eval_mode=False):
        if (not eval_mode) and np.random.rand() < 0.1:
            return np.random.randint(self.action_dim)
        x = torch.tensor(obs, dtype=torch.float32, device=self.device).view(1, 1, -1)
        with torch.no_grad():
            q, _ = self.q(x)
        return int(q[0, -1].argmax().item())

    def update(self, s, a, r, ns, d):
        self.buf.add((s, a, r, ns, d))
        if len(self.buf) < self.batch:
            return None
        batch, idx, w = self.buf.sample(self.batch, beta=self.beta)
        s, a, r, ns, d = map(np.array, zip(*batch))
        s = torch.tensor(s, dtype=torch.float32, device=self.device).view(self.batch, 1, -1)
        ns = torch.tensor(ns, dtype=torch.float32, device=self.device).view(self.batch, 1, -1)
        a = torch.tensor(a, dtype=torch.long, device=self.device)
        r = torch.tensor(r, dtype=torch.float32, device=self.device)
        d = torch.tensor(d, dtype=torch.float32, device=self.device)
        w = torch.tensor(w, dtype=torch.float32, device=self.device)
        q, _ = self.q(s)
        q = q[:, -1].gather(1, a.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            nq_online, _ = self.q(ns)
            na = nq_online[:, -1].argmax(-1)
            nq_tgt, _ = self.tgt(ns)
            y = r + self.gamma * (1 - d) * nq_tgt[:, -1].gather(1, na.unsqueeze(-1)).squeeze(-1)
        td = y - q
        loss = (w * td.pow(2)).mean()
        self.opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(self.q.parameters(), 40.0); self.opt.step()
        self.buf.update(idx, td.detach().cpu().numpy())
        return float(loss.item())
