"""PPO agent with clipped surrogate objective."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.distributions import Categorical

from agents.ppo.actor_critic import ActorCritic
from agents.ppo.rollout_buffer import RolloutBuffer


@dataclass
class PpoConfig:
    obs_dim:    int   = 18
    action_dim: int   = 27
    lr:         float = 3e-4
    gamma:      float = 0.99
    lambda_gae: float = 0.95
    eps_clip:   float = 0.2
    n_steps:    int   = 1000   # <= max_steps so update fires every episode
    epochs:     int   = 4
    batch_size: int   = 64
    ent_coef:   float = 0.01
    vf_coef:    float = 0.5
    grad_clip:  float = 0.5


class PpoAgent:
    def __init__(self, cfg: PpoConfig, device: torch.device) -> None:
        self.cfg    = cfg
        self.device = device
        self.net    = ActorCritic(cfg.obs_dim, cfg.action_dim).to(device)
        self.optim  = optim.Adam(self.net.parameters(), lr=cfg.lr)
        self.buffer = RolloutBuffer()

    @torch.no_grad()
    def act(self, obs: np.ndarray, eval_mode: bool = False):
        """Return (action, log_prob, value)."""
        t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, v = self.net(t)
        dist = Categorical(logits=logits)
        a    = dist.probs.argmax(-1) if eval_mode else dist.sample()
        return int(a.item()), float(dist.log_prob(a).item()), float(v.squeeze().item())

    def update(self, last_value: float = 0.0) -> Dict[str, float]:
        """Run PPO epochs on the current buffer, then clear it."""
        if len(self.buffer) == 0:
            return {"loss": 0.0}

        obs_np, actions, log_probs_old, rewards, dones, values = self.buffer.get()

        obs_t        = torch.tensor(obs_np,       dtype=torch.float32, device=self.device)
        actions_t    = torch.tensor(actions,      dtype=torch.long,    device=self.device)
        old_lp_t     = torch.tensor(log_probs_old,dtype=torch.float32, device=self.device)
        values_all_t = torch.tensor(values + [last_value],
                                    dtype=torch.float32, device=self.device)

        # GAE advantage
        adv_list, gae = [], 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.cfg.gamma * values_all_t[t + 1] * (1.0 - dones[t]) - values_all_t[t]
            gae   = delta + self.cfg.gamma * self.cfg.lambda_gae * (1.0 - dones[t]) * gae
            adv_list.append(gae)
        adv = torch.tensor(list(reversed(adv_list)), dtype=torch.float32, device=self.device)
        ret = adv + values_all_t[:-1]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        total_loss = 0.0
        n_updates  = 0

        for _ in range(self.cfg.epochs):
            idx = torch.randperm(len(obs_t), device=self.device)
            for i in range(0, len(obs_t), self.cfg.batch_size):
                b = idx[i:i + self.cfg.batch_size]
                logits, vpred = self.net(obs_t[b])
                dist   = Categorical(logits=logits)
                nlp    = dist.log_prob(actions_t[b])
                ratio  = torch.exp(nlp - old_lp_t[b])
                surr1  = ratio * adv[b]
                surr2  = torch.clamp(ratio, 1.0 - self.cfg.eps_clip,
                                            1.0 + self.cfg.eps_clip) * adv[b]
                p_loss = -torch.min(surr1, surr2).mean()
                v_loss = F.mse_loss(vpred.squeeze(-1), ret[b])
                e_loss = dist.entropy().mean()
                loss   = p_loss + self.cfg.vf_coef * v_loss - self.cfg.ent_coef * e_loss

                self.optim.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.grad_clip)
                self.optim.step()

                total_loss += float(loss.item())
                n_updates  += 1

        self.buffer.clear()
        return {"loss": total_loss / max(1, n_updates)}

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model": self.net.state_dict(),
            "optim": self.optim.state_dict(),
            "cfg":   self.cfg.__dict__,
        }, path)
