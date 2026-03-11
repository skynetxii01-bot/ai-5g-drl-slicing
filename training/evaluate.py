"""Evaluation script for trained slice allocation agents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from agents.dqn.dqn_agent import DQNAgent
from agents.ppo.ppo_agent import PPOAgent
from agents.r2d2.r2d2_agent import R2D2Agent
from envs.slice_gym_env import SliceGymEnv


def load_best_model(agent_name: str, agent):
    ckpts = sorted(Path("results/models").glob(f"{agent_name}_ep*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found for {agent_name}")
    best = ckpts[-1]
    net = getattr(agent, "online", getattr(agent, "model"))
    net.load_state_dict(torch.load(best, map_location="cpu"))
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--agent", choices=["dqn", "ppo", "r2d2"], default="dqn")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = SliceGymEnv(port=args.port, seed=args.seed)
    agent = {"dqn": DQNAgent(), "ppo": PPOAgent(), "r2d2": R2D2Agent()}[args.agent]
    load_best_model(args.agent, agent)

    rewards, embb_thr, urllc_lat, sla_pct = [], [], [], []
    for _ in range(50):
        obs = env.reset()
        done = False
        total = 0.0
        if args.agent == "r2d2":
            agent.reset_hidden()
        while not done:
            if args.agent == "ppo":
                action, _, _ = agent.select_action(obs, eval_mode=True)
            else:
                action = agent.select_action(obs, eval_mode=True)
            obs, reward, done, info = env.step(action)
            total += reward
            embb_thr.append(float(info["throughput_norm"][0]))
            urllc_lat.append(float(info["latency_norm"][1]))
            sla_pct.append(float(np.mean(info["throughput_norm"] > 0.8)))
        rewards.append(total)

    def ms(v):
        return {"mean": float(np.mean(v)), "std": float(np.std(v))}

    out = {
        "reward": ms(rewards),
        "embb_thr": ms(embb_thr),
        "urllc_lat": ms(urllc_lat),
        "sla_pct": ms(sla_pct),
    }
    Path("results").mkdir(exist_ok=True)
    with open("results/evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    env.close()


if __name__ == "__main__":
    main()
