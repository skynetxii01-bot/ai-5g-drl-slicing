"""
agents/ppo/ppo_agent.py
========================
Proximal Policy Optimization agent.
lr=3e-4, clip=0.2, n_steps=2048, epochs=4, GAE λ=0.95
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .actor_critic import ActorCritic
from .rollout_buffer import RolloutBuffer


class PPOAgent:
    def __init__(self, obs_dim=15, n_actions=27, lr=3e-4, gamma=0.99,
                 lam=0.95, clip=0.2, n_steps=2048, epochs=4,
                 batch_size=64, device=None):
        self.clip       = clip
        self.epochs     = epochs
        self.n_steps    = n_steps
        self.batch_size = batch_size
        self.device     = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.net       = ActorCritic(obs_dim, n_actions).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr, eps=1e-5)
        self.buffer    = RolloutBuffer(n_steps, obs_dim, gamma, lam, self.device)
        self.total_steps = 0

        print(f"PPOAgent on {self.device}. Params: {sum(p.numel() for p in self.net.parameters()):,}")

    def select_action(self, obs):
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, log_prob, _, value = self.net.get_action(obs_t)
        return action.item(), log_prob.item(), value.item()

    def store(self, obs, action, reward, value, log_prob, done):
        self.buffer.add(obs, action, reward,
                        torch.tensor(value), log_prob, done)
        self.total_steps += 1

    def train(self, last_obs):
        """Update policy using collected rollout."""
        last_obs_t = torch.FloatTensor(last_obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, last_value = self.net(last_obs_t)
        last_value = last_value.item()

        advantages, returns = self.buffer.compute_returns(last_value)
        obs, actions, old_log_probs = self.buffer.get()

        # Normalise advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = 0
        for _ in range(self.epochs):
            # Mini-batch updates
            idx = np.random.permutation(len(obs))
            for start in range(0, len(obs), self.batch_size):
                mb_idx   = torch.LongTensor(idx[start:start+self.batch_size])
                mb_obs   = obs[mb_idx]
                mb_act   = actions[mb_idx]
                mb_adv   = advantages[mb_idx]
                mb_ret   = returns[mb_idx]
                mb_oldlp = old_log_probs[mb_idx]

                new_log_probs, entropy, values = self.net.evaluate(mb_obs, mb_act)
                ratio = (new_log_probs - mb_oldlp).exp()

                # Clipped surrogate objective
                surr1 = ratio * mb_adv
                surr2 = ratio.clamp(1 - self.clip, 1 + self.clip) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss   = 0.5 * (mb_ret - values).pow(2).mean()
                entropy_loss = -0.01 * entropy.mean()

                loss = policy_loss + value_loss + entropy_loss
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.optimizer.step()
                total_loss += loss.item()

        self.buffer.reset()
        return total_loss

    def save(self, path):
        torch.save({'net': self.net.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'total_steps': self.total_steps}, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt['net'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.total_steps = ckpt.get('total_steps', 0)
