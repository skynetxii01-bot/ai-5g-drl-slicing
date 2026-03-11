"""
training/train.py
==================
Main training script for all DRL agents.

Usage:
  python3 training/train.py --agent dqn   --port 5555 --seed 42 --episodes 500
  python3 training/train.py --agent ppo   --port 5555 --seed 42 --episodes 500
  python3 training/train.py --agent r2d2  --port 5555 --seed 42 --episodes 500

NOTE: NS-3 simulation must be running FIRST in another terminal:
  ./ns3 run "scratch/slice-rl/slice-rl-sim --gymPort=5555 --seed=42"
"""

import argparse
import json
import os
import sys
import time
import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.slice_gym_env import SliceGymEnv
from agents.dqn.dqn_agent  import DQNAgent
from agents.ppo.ppo_agent  import PPOAgent
from agents.r2d2.r2d2_agent import R2D2Agent

# ── Directories ──────────────────────────────────────────────────────────────
os.makedirs('results/models', exist_ok=True)
os.makedirs('results/logs',   exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser(description='Train DRL agent for 5G NR slicing')
    p.add_argument('--agent',    type=str,   default='dqn',
                   choices=['dqn', 'ppo', 'r2d2'])
    p.add_argument('--port',     type=int,   default=5555)
    p.add_argument('--seed',     type=int,   default=42)
    p.add_argument('--episodes', type=int,   default=500)
    p.add_argument('--save_every', type=int, default=50)
    return p.parse_args()


def make_agent(name, device):
    """Instantiate the requested agent."""
    if name == 'dqn':
        return DQNAgent(obs_dim=15, n_actions=27, device=device)
    elif name == 'ppo':
        return PPOAgent(obs_dim=15, n_actions=27, device=device)
    elif name == 'r2d2':
        return R2D2Agent(obs_dim=15, n_actions=27, device=device)
    raise ValueError(f"Unknown agent: {name}")


def log_episode(log_path, episode, metrics):
    """Append one episode result to JSONL log file."""
    with open(log_path, 'a') as f:
        f.write(json.dumps({'episode': episode, **metrics}) + '\n')


def run_baseline(env, policy_fn, n_episodes=20):
    """Run a baseline policy and return average metrics."""
    all_rewards, all_sla = [], []
    for _ in range(n_episodes):
        obs = env.reset()
        total_reward, sla_sum, steps = 0, 0, 0
        done = False
        while not done:
            action = policy_fn(obs)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            sla_sum      += env.compute_sla_compliance(obs)
            steps        += 1
        all_rewards.append(total_reward)
        all_sla.append(sla_sum / max(steps, 1))
    return np.mean(all_rewards), np.mean(all_sla)


def train_dqn(agent, env, args, log_path):
    """DQN training loop."""
    best_reward = -np.inf

    for ep in range(1, args.episodes + 1):
        obs   = env.reset()
        total_reward = 0
        sla_sum, thr_sum, lat_sum = 0, 0, 0
        steps = 0
        done  = False

        while not done:
            action = agent.select_action(obs)
            next_obs, reward, done, _ = env.step(action)

            agent.store(obs, action, reward, next_obs, done)
            agent.train_step()

            decoded = env.decode_obs(next_obs)
            total_reward += reward
            thr_sum      += decoded['thr_mbps'].mean()
            lat_sum      += decoded['lat_ms'].mean()
            sla_sum      += env.compute_sla_compliance(next_obs)
            steps        += 1
            obs = next_obs

        metrics = {
            'reward':   round(total_reward, 3),
            'sla_pct':  round(sla_sum / max(steps, 1) * 100, 1),
            'avg_thr':  round(thr_sum / max(steps, 1), 2),
            'avg_lat':  round(lat_sum / max(steps, 1), 2),
            'eps':      round(agent.eps, 3),
            'steps':    steps,
        }
        log_episode(log_path, ep, metrics)

        if ep % 10 == 0:
            print(f"[DQN] Ep {ep:4d}/{args.episodes}  "
                  f"R={metrics['reward']:+.2f}  "
                  f"SLA={metrics['sla_pct']:.1f}%  "
                  f"ε={metrics['eps']:.3f}")

        if total_reward > best_reward:
            best_reward = total_reward
            agent.save(f'results/models/dqn_best.pt')

        if ep % args.save_every == 0:
            agent.save(f'results/models/dqn_ep{ep}.pt')

    print(f"[DQN] Training complete. Best reward: {best_reward:.3f}")


def train_ppo(agent, env, args, log_path):
    """PPO training loop."""
    best_reward = -np.inf
    ep = 0

    while ep < args.episodes:
        obs   = env.reset()
        done  = False
        step_count = 0
        ep_reward  = 0

        while not done:
            action, log_prob, value = agent.select_action(obs)
            next_obs, reward, done, _ = env.step(action)

            agent.store(obs, action, reward, value, log_prob, done)
            ep_reward += reward
            step_count += 1
            obs = next_obs

            if step_count % agent.n_steps == 0:
                agent.train(obs)

        ep += 1
        metrics = {'reward': round(ep_reward, 3), 'steps': step_count}
        log_episode(log_path, ep, metrics)

        if ep % 10 == 0:
            print(f"[PPO] Ep {ep:4d}/{args.episodes}  R={ep_reward:+.2f}")

        if ep_reward > best_reward:
            best_reward = ep_reward
            agent.save('results/models/ppo_best.pt')

        if ep % args.save_every == 0:
            agent.save(f'results/models/ppo_ep{ep}.pt')

    print(f"[PPO] Training complete. Best reward: {best_reward:.3f}")


def train_r2d2(agent, env, args, log_path):
    """R2D2 training loop."""
    best_reward = -np.inf

    for ep in range(1, args.episodes + 1):
        obs   = env.reset()
        done  = False
        total_reward = 0
        steps = 0

        while not done:
            action = agent.select_action(obs, eps=max(0.05, 1.0 - ep/200))
            next_obs, reward, done, _ = env.step(action)
            agent.store(obs, action, reward, done)
            agent.train_step()
            total_reward += reward
            steps        += 1
            obs = next_obs

        metrics = {'reward': round(total_reward, 3), 'steps': steps}
        log_episode(log_path, ep, metrics)

        if ep % 10 == 0:
            print(f"[R2D2] Ep {ep:4d}/{args.episodes}  R={total_reward:+.2f}")

        if total_reward > best_reward:
            best_reward = total_reward
            agent.save('results/models/r2d2_best.pt')

        if ep % args.save_every == 0:
            agent.save(f'results/models/r2d2_ep{ep}.pt')

    print(f"[R2D2] Training complete. Best reward: {best_reward:.3f}")


def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"\n{'='*50}")
    print(f"  Training {args.agent.upper()} agent")
    print(f"  Port: {args.port}  Episodes: {args.episodes}  Seed: {args.seed}")
    print(f"  Device: {device}")
    print(f"{'='*50}\n")

    print("Connecting to NS-3 simulation...")
    env      = SliceGymEnv(port=args.port, seed=args.seed)
    agent    = make_agent(args.agent, device)
    log_path = f'results/logs/{args.agent}_log.jsonl'

    # Run baselines first for comparison
    print("\nRunning baselines (5 episodes each)...")
    rnd_r, rnd_sla = run_baseline(env, env.random_policy, n_episodes=5)
    rr_r,  rr_sla  = run_baseline(env, env.round_robin_policy, n_episodes=5)
    pf_r,  pf_sla  = run_baseline(env, env.proportional_fair_policy, n_episodes=5)
    print(f"  Random:        R={rnd_r:+.2f}  SLA={rnd_sla*100:.1f}%")
    print(f"  Round-Robin:   R={rr_r:+.2f}  SLA={rr_sla*100:.1f}%")
    print(f"  Prop. Fair:    R={pf_r:+.2f}  SLA={pf_sla*100:.1f}%\n")

    # Save baseline results
    with open(f'results/logs/baselines.json', 'w') as f:
        json.dump({
            'random':      {'reward': rnd_r, 'sla_pct': rnd_sla * 100},
            'round_robin': {'reward': rr_r,  'sla_pct': rr_sla  * 100},
            'prop_fair':   {'reward': pf_r,  'sla_pct': pf_sla  * 100},
        }, f, indent=2)

    # Train the chosen agent
    start = time.time()
    if   args.agent == 'dqn':  train_dqn (agent, env, args, log_path)
    elif args.agent == 'ppo':  train_ppo (agent, env, args, log_path)
    elif args.agent == 'r2d2': train_r2d2(agent, env, args, log_path)

    elapsed = time.time() - start
    print(f"\nTotal training time: {elapsed/3600:.2f} hours")
    env.close()


if __name__ == '__main__':
    main()
