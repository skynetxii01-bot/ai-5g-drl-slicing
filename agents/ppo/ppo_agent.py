"""PPO agent implementation for discrete action selection."""

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.distributions import Categorical

from agents.ppo.actor_critic import ActorCritic
from agents.ppo.rollout_buffer import RolloutBuffer


class PPOAgent:
    def __init__(self, obs_dim: int = 15, act_dim: int = 27, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = ActorCritic(obs_dim, act_dim).to(self.device)
        self.optim = optim.Adam(self.model.parameters(), lr=3e-4)
        self.buffer = RolloutBuffer()
        self.gamma = 0.99
        self.lam = 0.95
        self.clip = 0.2
        self.n_steps = 2048
        self.epochs = 4

    def select_action(self, obs: np.ndarray, eval_mode: bool = False):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits, value = self.model(obs_t)
            dist = Categorical(logits=logits)
            if eval_mode:
                action = torch.argmax(logits, dim=1)
            else:
                action = dist.sample()
            logprob = dist.log_prob(action)
        return int(action.item()), float(value.item()), float(logprob.item())

    def store(self, obs, action, reward, done, value, logprob):
        self.buffer.add(obs, action, reward, done, value, logprob)

    def ready(self):
        return len(self.buffer.rewards) >= self.n_steps

    def update(self, last_obs):
        obs_t = torch.tensor(last_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            _, last_value = self.model(obs_t)

        returns, adv = self.buffer.compute_gae(float(last_value.item()), self.gamma, self.lam)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs = torch.tensor(np.array(self.buffer.obs), dtype=torch.float32, device=self.device)
        actions = torch.tensor(self.buffer.actions, dtype=torch.long, device=self.device)
        old_logprob = torch.tensor(self.buffer.logprobs, dtype=torch.float32, device=self.device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        adv_t = torch.tensor(adv, dtype=torch.float32, device=self.device)

        for _ in range(self.epochs):
            logits, values = self.model(obs)
            dist = Categorical(logits=logits)
            new_logprob = dist.log_prob(actions)
            ratio = (new_logprob - old_logprob).exp()

            surr1 = ratio * adv_t
            surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv_t
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(values.squeeze(-1), returns_t)
            entropy = dist.entropy().mean()
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

        self.buffer.clear()
