"""R2D2-inspired agent with sequence replay and recurrent hidden state."""

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from agents.r2d2.per_buffer import PrioritizedSequenceReplay
from agents.r2d2.r2d2_network import R2D2Network


class R2D2Agent:
    def __init__(self, obs_dim=15, act_dim=27, device="cpu"):
        self.device = torch.device(device)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.seq_len = 40
        self.burn_in = 40
        self.n_step = 5
        self.gamma = 0.99
        self.total_steps = 0

        self.online = R2D2Network(obs_dim, act_dim).to(self.device)
        self.target = R2D2Network(obs_dim, act_dim).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.optim = optim.Adam(self.online.parameters(), lr=1e-4)
        self.buffer = PrioritizedSequenceReplay(alpha=0.6, beta_start=0.4, beta_end=1.0)
        self.hidden = None

    def reset_hidden(self):
        self.hidden = None

    def select_action(self, obs, eval_mode=False):
        x = torch.tensor(obs, dtype=torch.float32, device=self.device).view(1, 1, -1)
        with torch.no_grad():
            q, self.hidden = self.online(x, self.hidden)
            action = int(torch.argmax(q[0, 0]).item())
        if (not eval_mode) and np.random.rand() < max(0.05, 1.0 - self.total_steps / 50000):
            action = np.random.randint(0, self.act_dim)
        self.total_steps += 1
        return action

    def store_sequence(self, sequence):
        self.buffer.add(sequence)

    def update(self, batch_size=8):
        if len(self.buffer) < batch_size:
            return None
        batch, idxs, is_weights = self.buffer.sample(batch_size, self.total_steps, 100000)
        losses = []
        td_errs = []
        for seq in batch:
            states = torch.tensor(np.array([x[0] for x in seq]), dtype=torch.float32, device=self.device).unsqueeze(0)
            actions = torch.tensor([x[1] for x in seq], dtype=torch.long, device=self.device)
            rewards = torch.tensor([x[2] for x in seq], dtype=torch.float32, device=self.device)
            next_states = torch.tensor(np.array([x[3] for x in seq]), dtype=torch.float32, device=self.device).unsqueeze(0)
            dones = torch.tensor([x[4] for x in seq], dtype=torch.float32, device=self.device)

            q, _ = self.online(states)
            q = q.squeeze(0).gather(1, actions.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                tq, _ = self.target(next_states)
                tq = tq.squeeze(0).max(dim=1)[0]
                target = rewards + self.gamma * (1 - dones) * tq
            loss = F.mse_loss(q, target)
            losses.append(loss)
            td_errs.append(torch.mean(torch.abs(q - target)).item())

        w = torch.tensor(is_weights, dtype=torch.float32, device=self.device)
        total_loss = torch.mean(torch.stack(losses) * w)
        self.optim.zero_grad()
        total_loss.backward()
        self.optim.step()
        self.buffer.update_priorities(idxs, td_errs)

        for tp, op in zip(self.target.parameters(), self.online.parameters()):
            tp.data.copy_(0.995 * tp.data + 0.005 * op.data)
        return float(total_loss.item())
