"""R2D2 agent — recurrent Double-DQN with Prioritized Experience Replay."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from agents.r2d2.per_buffer import PrioritizedReplayBuffer
from agents.r2d2.r2d2_network import R2D2Net


@dataclass
class R2D2Config:
    obs_dim:       int   = 18
    action_dim:    int   = 27
    lr:            float = 1e-4
    gamma:         float = 0.99
    batch_size:    int   = 32
    seq_len:       int   = 80    # total sequence stored (gradient window = seq_len - burn_in)
    burn_in:       int   = 40    # LSTM warm-up steps — no gradient here
    
    per_alpha:     float = 0.6
    per_beta_start:float = 0.4
    per_beta_end:  float = 1.0
    buffer_size:   int   = 100_000
    grad_clip:     float = 40.0
    eps_explore:   float = 0.1   # fixed epsilon for R2D2 (no decay schedule)


class R2D2Agent:
    def __init__(self, cfg: R2D2Config, device: torch.device) -> None:
        self.cfg    = cfg
        self.device = device

        self.online = R2D2Net(cfg.obs_dim, cfg.action_dim).to(device)
        self.target = R2D2Net(cfg.obs_dim, cfg.action_dim).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optim  = optim.Adam(self.online.parameters(), lr=cfg.lr)
        self.buffer = PrioritizedReplayBuffer(cfg.buffer_size, alpha=cfg.per_alpha)

        self.train_steps = 0

    def act(
        self,
        obs:     np.ndarray,
        hidden:  Optional[Tuple] = None,
        explore: bool = True,
    ) -> Tuple[int, Optional[Tuple]]:
        """Return (action, hidden_state). Epsilon-greedy with fixed epsilon."""
        if explore and np.random.rand() < self.cfg.eps_explore:
            return int(np.random.randint(self.cfg.action_dim)), hidden

        x = torch.tensor(obs, dtype=torch.float32, device=self.device).view(1, 1, -1)
        with torch.no_grad():
            q, hidden_out = self.online(x, hidden)
        return int(q[0, -1].argmax().item()), hidden_out

    def add_sequence(
        self,
        obs:      np.ndarray,   # (seq_len, obs_dim)
        action:   np.ndarray,   # (seq_len,)
        reward:   np.ndarray,   # (seq_len,)
        done:     np.ndarray,   # (seq_len,)
        next_obs: np.ndarray,   # (seq_len, obs_dim)
        priority: float = 1.0,
    ) -> None:
        """Pack arrays into a dict and store in PER buffer."""
        seq = dict(obs=obs, action=action, reward=reward,
                   done=done, next_obs=next_obs)
        self.buffer.add(seq, priority)

    def train_step(self) -> Dict[str, float]:
        """Sample a batch of sequences, run burn-in, compute loss, update."""
        if len(self.buffer) < self.cfg.batch_size:
            return {"loss": 0.0}

        # Beta annealing toward 1.0 over 50k train steps
        beta = min(
            self.cfg.per_beta_end,
            self.cfg.per_beta_start
            + self.train_steps
            * (self.cfg.per_beta_end - self.cfg.per_beta_start)
            / 50_000,
        )

        seqs, idx, weights = self.buffer.sample(self.cfg.batch_size, beta=beta)

        # Stack into tensors — shape (batch, seq_len, ...)
        obs_t  = torch.tensor(np.stack([s["obs"]      for s in seqs]),
                               dtype=torch.float32, device=self.device)
        nxt_t  = torch.tensor(np.stack([s["next_obs"] for s in seqs]),
                               dtype=torch.float32, device=self.device)
        act_t  = torch.tensor(np.stack([s["action"]   for s in seqs]),
                               dtype=torch.long,    device=self.device)
        rew_t  = torch.tensor(np.stack([s["reward"]   for s in seqs]),
                               dtype=torch.float32, device=self.device)
        done_t = torch.tensor(np.stack([s["done"]     for s in seqs]),
                               dtype=torch.float32, device=self.device)
        w_t    = torch.tensor(weights, dtype=torch.float32, device=self.device)

        bi = self.cfg.burn_in   # gradient window starts here

        # ── Burn-in: warm up LSTM without gradient ────────────────────
        with torch.no_grad():
            _, hc_on  = self.online(obs_t[:, :bi], None)
            _, hc_tgt = self.target(obs_t[:, :bi], None)

        hc_on  = tuple(h.detach() for h in hc_on)
        hc_tgt = tuple(h.detach() for h in hc_tgt)

        # ── Gradient window: [burn_in : seq_len] ──────────────────────
        q_all, _ = self.online(obs_t[:, bi:], hc_on)   # (B, grad_steps, A)

        with torch.no_grad():
            nq_on,  _ = self.online(nxt_t[:, bi:], hc_on)
            nq_tgt, _ = self.target(nxt_t[:, bi:], hc_tgt)

        # Double DQN: select action with online, evaluate with target
        na    = nq_on.argmax(dim=-1, keepdim=True)              # (B, G, 1)
        nq    = nq_tgt.gather(-1, na).squeeze(-1)               # (B, G)
        q     = q_all.gather(-1, act_t[:, bi:].unsqueeze(-1)).squeeze(-1)  # (B, G)
        y     = rew_t[:, bi:] + self.cfg.gamma * (1.0 - done_t[:, bi:]) * nq

        td    = (y.detach() - q)
        loss  = (w_t.unsqueeze(1) *
                 F.smooth_l1_loss(q, y.detach(), reduction="none")).mean()

        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), self.cfg.grad_clip)
        self.optim.step()
        self.train_steps += 1

        # Update PER priorities with mean |TD error| per sequence
        td_mean = td.abs().mean(dim=1).detach().cpu().numpy()
        self.buffer.update(idx, td_mean)

        return {"loss": float(loss.item())}

    def sync_target(self) -> None:
        """Hard update: copy online weights to target."""
        self.target.load_state_dict(self.online.state_dict())

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "online":      self.online.state_dict(),
            "target":      self.target.state_dict(),
            "optim":       self.optim.state_dict(),
            "train_steps": self.train_steps,
            "cfg":         self.cfg.__dict__,
        }, path)
