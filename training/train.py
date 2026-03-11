"""Training entrypoint for DQN/PPO/R2D2 on NS-3 slice environment."""

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


def make_agent(name: str, device: str):
    if name == "dqn":
        return DQNAgent(device=device)
    if name == "ppo":
        return PPOAgent(device=device)
    if name == "r2d2":
        return R2D2Agent(device=device)
    raise ValueError(name)


def run_baseline(env: SliceGymEnv, policy_name: str, episodes: int = 5):
    rewards = []
    for _ in range(episodes):
        obs = env.reset()
        done = False
        total = 0.0
        while not done:
            if policy_name == "random":
                action = env.random_policy(obs)
            elif policy_name == "round_robin":
                action = env.round_robin_policy(obs)
            else:
                action = env.proportional_fair_policy(obs)
            obs, r, done, _ = env.step(action)
            total += r
        rewards.append(total)
    return float(np.mean(rewards))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=["dqn", "ppo", "r2d2"], required=True)
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=500)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    models_dir = Path("results/models")
    logs_dir = Path("results/logs")
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{args.agent}_log.jsonl"

    env = SliceGymEnv(port=args.port, seed=args.seed)
    agent = make_agent(args.agent, "cpu")

    baselines = {
        "random": run_baseline(env, "random"),
        "round_robin": run_baseline(env, "round_robin"),
        "proportional_fair": run_baseline(env, "proportional_fair"),
    }

    with log_path.open("w", encoding="utf-8") as f:
        for name, score in baselines.items():
            f.write(json.dumps({"episode": -1, "policy": name, "reward": score}) + "\n")

    for ep in range(1, args.episodes + 1):
        obs = env.reset()
        done = False
        total_reward = 0.0
        embb_thr_vals, urllc_lat_vals, sla_vals = [], [], []
        seq = []
        if args.agent == "r2d2":
            agent.reset_hidden()

        while not done:
            if args.agent == "ppo":
                action, value, logprob = agent.select_action(obs)
            else:
                action = agent.select_action(obs)

            next_obs, reward, done, info = env.step(action)
            total_reward += reward
            embb_thr_vals.append(float(info["throughput_norm"][0]))
            urllc_lat_vals.append(float(info["latency_norm"][1]))
            sla_vals.append(float(np.mean(info["throughput_norm"] > 0.8)))

            if args.agent == "dqn":
                agent.store(obs, action, reward, next_obs, done)
                agent.update()
            elif args.agent == "ppo":
                agent.store(obs, action, reward, done, value, logprob)
                if agent.ready():
                    agent.update(next_obs)
            else:
                seq.append((obs, action, reward, next_obs, done))
                if len(seq) >= agent.seq_len:
                    agent.store_sequence(seq[-agent.seq_len :])
                    agent.update()

            obs = next_obs

        if args.agent == "ppo" and len(agent.buffer.rewards) > 0:
            agent.update(obs)

        rec = {
            "episode": ep,
            "reward": total_reward,
            "embb_thr": float(np.mean(embb_thr_vals) if embb_thr_vals else 0.0),
            "urllc_lat": float(np.mean(urllc_lat_vals) if urllc_lat_vals else 0.0),
            "sla_pct": float(np.mean(sla_vals) if sla_vals else 0.0),
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

        if ep % 50 == 0:
            torch.save(getattr(agent, "online", getattr(agent, "model")).state_dict(), models_dir / f"{args.agent}_ep{ep}.pt")
            print(f"Saved checkpoint at episode {ep}")

    env.close()


if __name__ == "__main__":
    main()
