import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.distributions import Categorical

from agents.ppo.actor_critic import ActorCritic
from agents.ppo.rollout_buffer import RolloutBuffer


class PPOAgent:
    def __init__(self, state_dim=15, action_dim=27, device="cpu"):
        self.device = torch.device(device)
        self.net = ActorCritic(state_dim, action_dim).to(self.device)
        self.opt = optim.Adam(self.net.parameters(), lr=3e-4)
        self.buf = RolloutBuffer()
        self.gamma, self.lam = 0.99, 0.95
        self.clip, self.epochs, self.batch, self.ent_coef = 0.2, 4, 64, 0.01

    def act(self, obs, eval_mode=False):
        t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, v = self.net(t)
        dist = Categorical(logits=logits)
        a = dist.probs.argmax(-1) if eval_mode else dist.sample()
        return int(a.item()), float(dist.log_prob(a).item()), float(v.item())

    def update(self, last_value=0.0):
        if not self.buf.data:
            return None
        s, a, lp, r, d, v = map(list, zip(*self.buf.data))
        s = torch.tensor(np.array(s), dtype=torch.float32, device=self.device)
        a = torch.tensor(a, dtype=torch.long, device=self.device)
        old_lp = torch.tensor(lp, dtype=torch.float32, device=self.device)
        vals = torch.tensor(v + [last_value], dtype=torch.float32, device=self.device)
        adv, gae = [], 0.0
        for t in reversed(range(len(r))):
            delta = r[t] + self.gamma * vals[t + 1] * (1 - d[t]) - vals[t]
            gae = delta + self.gamma * self.lam * (1 - d[t]) * gae
            adv.append(gae)
        adv = torch.tensor(list(reversed(adv)), dtype=torch.float32, device=self.device)
        ret = adv + vals[:-1]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        for _ in range(self.epochs):
            idx = torch.randperm(len(s), device=self.device)
            for i in range(0, len(s), self.batch):
                b = idx[i:i + self.batch]
                logits, vpred = self.net(s[b])
                dist = Categorical(logits=logits)
                nlp = dist.log_prob(a[b])
                ratio = torch.exp(nlp - old_lp[b])
                surr1 = ratio * adv[b]
                surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv[b]
                loss = -torch.min(surr1, surr2).mean() + 0.5 * F.mse_loss(vpred.squeeze(-1), ret[b]) - self.ent_coef * dist.entropy().mean()
                self.opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(self.net.parameters(), 10.0); self.opt.step()
        self.buf.clear()
        return True
