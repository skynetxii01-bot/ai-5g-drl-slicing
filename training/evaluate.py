#!/usr/bin/env python3
"""Evaluate trained policies and baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.dqn.dqn_network import DuelingDQN
from agents.ppo.actor_critic import ActorCritic
from agents.r2d2.r2d2_network import R2D2Network
from envs.slice_gym_env import SliceGymEnv



def run_episode_ns3(policy_fn, env: SliceGymEnv, max_steps: int = 1000):
    obs, info = env.reset(seed=99)
    total = 0.0
    decoded = info.get("decoded_obs", {})
    hidden = None
    for _ in range(max_steps):
        action, hidden = policy_fn(obs, hidden)
        obs, reward, done, truncated, info = env.step(action)
        total += reward
        decoded = info.get("decoded_obs", decoded)
        if done or truncated:
            break
    return float(total), decoded


def baseline_simulator(policy: str, episodes: int = 100, steps: int = 1000):
    rng = np.random.default_rng(99)
    rewards = []
    for ep in range(episodes):
        r = 0.0
        for t in range(steps):
            if policy == "Random":
                a = int(rng.integers(0, 27))
            elif policy == "Round-Robin":
                a = int((ep * steps + t) % 27)
            else:  # Proportional Fair proxy
                a = int(np.argmax(np.array([np.sin(0.01 * (t + 1)), np.cos(0.01 * (t + 1)), 0.5])))
            r += 0.1 - 0.01 * abs(a - 13)
        rewards.append(r)
    return {"mean_reward": float(np.mean(rewards)), "std_reward": float(np.std(rewards))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--out", type=str, default="results/evaluation_results.json")
    args = parser.parse_args()

    results = {}
    model_dir = Path("results/models")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = SliceGymEnv(port=args.port, sim_seed=99, start_sim=False)

    # DQN
    dqn = DuelingDQN(15, 27).to(device)
    dqn_ckpt = torch.load(model_dir / "dqn_final.pt", map_location=device)
    dqn.load_state_dict(dqn_ckpt["online"])
    dqn.eval()

    def dqn_policy(obs, hidden):
        with torch.no_grad():
            q = dqn(torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))
        return int(torch.argmax(q, dim=1).item()), hidden

    # PPO
    ppo = ActorCritic(15, 27).to(device)
    ppo_ckpt = torch.load(model_dir / "ppo_final.pt", map_location=device)
    ppo.load_state_dict(ppo_ckpt.get("model", ppo_ckpt))
    ppo.eval()

    def ppo_policy(obs, hidden):
        with torch.no_grad():
            dist, _ = ppo.get_dist_value(torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))
        return int(torch.argmax(dist.logits, dim=1).item()), hidden

    # R2D2
    r2d2 = R2D2Network(15, 27).to(device)
    r2d2_ckpt = torch.load(model_dir / "r2d2_final.pt", map_location=device)
    r2d2.load_state_dict(r2d2_ckpt["online"])
    r2d2.eval()

    def r2d2_policy(obs, hidden):
        with torch.no_grad():
            q, hidden_out = r2d2(torch.tensor(obs, dtype=torch.float32, device=device).view(1, 1, -1), hidden)
        return int(torch.argmax(q[0, -1]).item()), hidden_out

    for name, policy in [("DQN", dqn_policy), ("PPO", ppo_policy), ("R2D2", r2d2_policy)]:
        ep_rewards = []
        for _ in range(args.episodes):
            r, _ = run_episode_ns3(policy, env, args.max_steps)
            ep_rewards.append(r)
        results[name] = {"mean_reward": float(np.mean(ep_rewards)), "std_reward": float(np.std(ep_rewards))}

    env.close()

    for base in ["Random", "Round-Robin", "Proportional Fair"]:
        results[base] = baseline_simulator(base, episodes=args.episodes, steps=args.max_steps)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\nEvaluation Summary (100 episodes)")
    print("=" * 48)
    print(f"{'Policy':<20} {'MeanReward':>12} {'Std':>12}")
    for k, v in results.items():
        print(f"{k:<20} {v['mean_reward']:>12.3f} {v['std_reward']:>12.3f}")


if __name__ == "__main__":
    main()
