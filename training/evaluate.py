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
from agents.r2d2.r2d2_network import R2D2Net
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


def run_ns3_baseline(policy: str, env: SliceGymEnv, episodes: int = 100, steps: int = 1000):
    rng = np.random.default_rng(99)
    rewards = []
    
    def pf_action_from_obs(obs: np.ndarray) -> int:
        # Greedy heuristic on observed normalized throughput shortfall.
        # obs[3:6] are throughput norms for [eMBB, URLLC, mMTC].
        thr = np.asarray(obs[3:6], dtype=np.float32)
        target_slice = int(np.argmin(thr))
        # Allocate +1 PRB to target slice, keep others unchanged.
        # Action index map uses [-1,0,1]^3 lexicographic order.
        deltas = [0, 0, 0]
        deltas[target_slice] = 1
        delta_to_action = {
            (-1, -1, -1): 0, (-1, -1, 0): 1, (-1, -1, 1): 2,
            (-1, 0, -1): 3, (-1, 0, 0): 4, (-1, 0, 1): 5,
            (-1, 1, -1): 6, (-1, 1, 0): 7, (-1, 1, 1): 8,
            (0, -1, -1): 9, (0, -1, 0): 10, (0, -1, 1): 11,
            (0, 0, -1): 12, (0, 0, 0): 13, (0, 0, 1): 14,
            (0, 1, -1): 15, (0, 1, 0): 16, (0, 1, 1): 17,
            (1, -1, -1): 18, (1, -1, 0): 19, (1, -1, 1): 20,
            (1, 0, -1): 21, (1, 0, 0): 22, (1, 0, 1): 23,
            (1, 1, -1): 24, (1, 1, 0): 25, (1, 1, 1): 26
        }
        return int(delta_to_action[tuple(deltas)])
    
    for ep in range(episodes):
        
        obs, _ = env.reset(seed=99 + ep)
        total = 0.0
        
        for t in range(steps):
            if policy == "Random":
                action = int(rng.integers(0, 27))
            elif policy == "Round-Robin":
                action = int((ep * steps + t) % 27)
            else:  # Proportional-fair-like throughput equalization heuristic
                action = pf_action_from_obs(obs)
            obs, reward, done, truncated, _ = env.step(action)
            total += float(reward)
            if done or truncated:
                break
        rewards.append(total)
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
            logits, _ = ppo(torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))
        return int(torch.argmax(logits, dim=1).item()), hidden

    # R2D2
    r2d2 = R2D2Net(15, 27).to(device)
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
    
    env_baseline = SliceGymEnv(port=args.port, sim_seed=199, start_sim=False)
    for base in ["Random", "Round-Robin", "Proportional Fair"]:
        results[base] = run_ns3_baseline(base, env_baseline, episodes=args.episodes, steps=args.max_steps)
    env_baseline.close()

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
