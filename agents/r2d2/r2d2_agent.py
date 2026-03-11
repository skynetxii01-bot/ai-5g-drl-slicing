"""
agents/r2d2/r2d2_agent.py
==========================
Recurrent Replay Distributed DQN (R2D2) agent.
LSTM hidden state persists → best for URLLC latency spikes.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .r2d2_network import R2D2Network
from .per_buffer import PERBuffer


class R2D2Agent:
    def __init__(self, obs_dim=15, n_actions=27, lr=1e-4, gamma=0.99,
                 seq_len=40, burn_in=40, n_step=5,
                 buffer_cap=10_000, batch_size=32, device=None):
        self.n_actions  = n_actions
        self.gamma      = gamma
        self.seq_len    = seq_len
        self.burn_in    = burn_in
        self.n_step     = n_step
        self.batch_size = batch_size
        self.device     = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.online_net = R2D2Network(obs_dim, n_actions).to(self.device)
        self.target_net = R2D2Network(obs_dim, n_actions).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer   = optim.Adam(self.online_net.parameters(), lr=lr)
        self.buffer      = PERBuffer(buffer_cap, seq_len, obs_dim, device=self.device)
        self.hidden      = self.online_net.init_hidden(device=str(self.device))
        self.total_steps = 0

        # Episode buffer for building sequences
        self._ep_obs = []
        self._ep_act = []
        self._ep_rew = []
        self._ep_don = []

        print(f"R2D2Agent on {self.device}. LSTM hidden=256, seq={seq_len}")

    def select_action(self, obs, eps=0.05):
        """Select action using LSTM network with persistent hidden state."""
        if np.random.random() < eps:
            return np.random.randint(self.n_actions)

        obs_t = torch.FloatTensor(obs).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q, self.hidden = self.online_net(obs_t, self.hidden)
        return q.squeeze().argmax().item()

    def store(self, obs, action, reward, done):
        """Accumulate episode steps; flush to buffer when episode ends."""
        self._ep_obs.append(obs)
        self._ep_act.append(action)
        self._ep_rew.append(reward)
        self._ep_don.append(float(done))
        self.total_steps += 1

        if done:
            self._flush_episode()
            # Reset LSTM hidden state at episode boundary
            self.hidden = self.online_net.init_hidden(device=str(self.device))

    def _flush_episode(self):
        """Break episode into overlapping sequences and store in PER buffer."""
        T = len(self._ep_obs)
        if T < self.seq_len:
            # Pad short episodes
            pad = self.seq_len - T
            self._ep_obs  += [np.zeros_like(self._ep_obs[0])] * pad
            self._ep_act  += [0] * pad
            self._ep_rew  += [0.0] * pad
            self._ep_don  += [1.0] * pad

        obs_arr = np.array(self._ep_obs[:self.seq_len])
        act_arr = np.array(self._ep_act[:self.seq_len])
        rew_arr = np.array(self._ep_rew[:self.seq_len])
        don_arr = np.array(self._ep_don[:self.seq_len])

        self.buffer.add(obs_arr, act_arr, rew_arr, don_arr)
        self._ep_obs, self._ep_act, self._ep_rew, self._ep_don = [], [], [], []

    def train_step(self):
        if len(self.buffer) < 100:
            return None

        obs, actions, rewards, dones, weights, idx = self.buffer.sample(self.batch_size)

        # Burn-in: run LSTM on first burn_in steps without gradient
        with torch.no_grad():
            _, hidden = self.online_net(obs[:, :self.burn_in], None)

        # Training on remaining steps
        q_online, _ = self.online_net(obs[:, self.burn_in:], hidden)
        q_target, _ = self.target_net(obs[:, self.burn_in:], hidden)

        T = q_online.size(1)
        loss_total = torch.tensor(0.0).to(self.device)
        td_errors  = np.zeros(self.batch_size)

        for t in range(T - self.n_step):
            q_val    = q_online[:, t].gather(1, actions[:, self.burn_in + t].unsqueeze(1)).squeeze()
            next_act = q_online[:, t + self.n_step].argmax(1)
            q_next   = q_target[:, t + self.n_step].gather(1, next_act.unsqueeze(1)).squeeze()

            # n-step return
            n_step_reward = sum(
                self.gamma**i * rewards[:, self.burn_in + t + i]
                for i in range(self.n_step)
            )
            q_tgt = n_step_reward + (self.gamma**self.n_step) * q_next * (1 - dones[:, self.burn_in + t])

            td_err   = (q_val - q_tgt.detach()).abs().detach().cpu().numpy()
            td_errors += td_err
            loss_total += (weights * (q_val - q_tgt.detach()).pow(2)).mean()

        self.optimizer.zero_grad()
        loss_total.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()

        # Update priorities
        self.buffer.update_priorities(idx, td_errors / T)

        # Hard update target every 1000 steps
        if self.total_steps % 1000 == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return loss_total.item()

    def save(self, path):
        torch.save({'online': self.online_net.state_dict(),
                    'target': self.target_net.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'total_steps': self.total_steps}, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ckpt['online'])
        self.target_net.load_state_dict(ckpt['target'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.total_steps = ckpt.get('total_steps', 0)
