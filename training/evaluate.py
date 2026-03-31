import json
from pathlib import Path

import numpy as np
import torch

from agents.dqn.dqn_agent import DQNAgent
from agents.ppo.ppo_agent import PPOAgent
from agents.r2d2.r2d2_agent import R2D2Agent
from envs.slice_gym_env import SliceGymEnv


def eval_policy(name, agent, env, episodes=100):
    rewards = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=99 + ep)
        done, total = False, 0.0
        while not done:
            if name == "ppo":
                a, _, _ = agent.act(obs, eval_mode=True)
            else:
                a = agent.act(obs, eval_mode=True)
            obs, r, d, t, _ = env.step(a)
            done = d or t
            total += r
        rewards.append(total)
    return {"mean_reward": float(np.mean(rewards)), "std_reward": float(np.std(rewards))}


def eval_baseline(kind, env, episodes=100):
    rewards = []
    rr = 0
    for ep in range(episodes):
        obs, _ = env.reset(seed=99 + ep)
        done, total = False, 0.0
        while not done:
            if kind == "Random":
                a = env.action_space.sample()
            elif kind == "RoundRobin":
                a = rr % 27; rr += 1
            else:
                a = 13  # neutral for PF proxy
            obs, r, d, t, _ = env.step(a)
            done = d or t
            total += r
        rewards.append(total)
    return {"mean_reward": float(np.mean(rewards)), "std_reward": float(np.std(rewards))}


def main():
    env = SliceGymEnv(port=5555, seed=99)
    models = Path("results/models")
    results = {}

    dqn = DQNAgent(); dqn.q.load_state_dict(torch.load(models / "dqn_final.pt", map_location="cpu"))
    ppo = PPOAgent(); ppo.net.load_state_dict(torch.load(models / "ppo_final.pt", map_location="cpu"))
    r2d2 = R2D2Agent(); r2d2.q.load_state_dict(torch.load(models / "r2d2_final.pt", map_location="cpu"))

    results["DQN"] = eval_policy("dqn", dqn, env)
    results["PPO"] = eval_policy("ppo", ppo, env)
    results["R2D2"] = eval_policy("r2d2", r2d2, env)
    results["Random"] = eval_baseline("Random", env)
    results["RoundRobin"] = eval_baseline("RoundRobin", env)
    results["ProportionalFair"] = eval_baseline("PF", env)

    Path("results").mkdir(exist_ok=True)
    with open("results/evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("Policy           MeanReward   Std")
    for k, v in results.items():
        print(f"{k:16} {v['mean_reward']:10.3f} {v['std_reward']:8.3f}")


if __name__ == "__main__":
    main()
