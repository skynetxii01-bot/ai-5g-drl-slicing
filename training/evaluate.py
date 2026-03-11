"""
training/evaluate.py
=====================
Evaluate saved model against baselines. Uses seed=99 (different from training).

Usage:
  Terminal 1: ./ns3 run "scratch/slice-rl/slice-rl-sim --gymPort=5555 --seed=99"
  Terminal 2: python3 training/evaluate.py --agent dqn --port 5555 --seed 99
"""
import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.slice_gym_env import SliceGymEnv
from agents.dqn.dqn_agent   import DQNAgent
from agents.ppo.ppo_agent   import PPOAgent
from agents.r2d2.r2d2_agent import R2D2Agent


def evaluate_policy(env, action_fn, n_episodes=50):
    all_rewards, all_sla, all_thr, all_lat = [], [], [], []

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        ep_reward, ep_sla, ep_thr, ep_lat, steps = 0, 0, 0, 0, 0

        while not done:
            action = action_fn(obs)
            obs, reward, done, _ = env.step(action)
            decoded = env.decode_obs(obs)
            ep_reward += reward
            ep_sla    += env.compute_sla_compliance(obs)
            ep_thr    += decoded['thr_mbps'].mean()
            ep_lat    += decoded['lat_ms'].mean()
            steps += 1

        all_rewards.append(ep_reward)
        all_sla.append(ep_sla / max(steps, 1))
        all_thr.append(ep_thr / max(steps, 1))
        all_lat.append(ep_lat / max(steps, 1))
        print(f"  Episode {ep+1:2d}: R={ep_reward:+.2f}  SLA={ep_sla/steps*100:.1f}%")

    return {
        'reward_mean': float(np.mean(all_rewards)),
        'reward_std':  float(np.std(all_rewards)),
        'sla_mean':    float(np.mean(all_sla) * 100),
        'thr_mean':    float(np.mean(all_thr)),
        'lat_mean':    float(np.mean(all_lat)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--agent',      type=str, default='dqn', choices=['dqn','ppo','r2d2'])
    p.add_argument('--port',       type=int, default=5555)
    p.add_argument('--seed',       type=int, default=99)
    p.add_argument('--n_episodes', type=int, default=50)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = SliceGymEnv(port=args.port, seed=args.seed)

    results = {}

    # Baselines
    print("\n--- Random Policy ---")
    results['random'] = evaluate_policy(env, env.random_policy, 10)

    print("\n--- Round-Robin Policy ---")
    results['round_robin'] = evaluate_policy(env, env.round_robin_policy, 10)

    print("\n--- Proportional Fair Policy ---")
    results['prop_fair'] = evaluate_policy(env, env.proportional_fair_policy, 10)

    # DRL agent
    print(f"\n--- {args.agent.upper()} Agent ---")
    model_path = f'results/models/{args.agent}_best.pt'
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        env.close()
        return

    if args.agent == 'dqn':
        agent = DQNAgent(device=device)
        agent.load(model_path)
        action_fn = lambda obs: agent.select_action(obs)
    elif args.agent == 'ppo':
        agent = PPOAgent(device=device)
        agent.load(model_path)
        action_fn = lambda obs: agent.select_action(obs)[0]
    elif args.agent == 'r2d2':
        agent = R2D2Agent(device=device)
        agent.load(model_path)
        action_fn = lambda obs: agent.select_action(obs, eps=0.0)

    results[args.agent] = evaluate_policy(env, action_fn, args.n_episodes)

    os.makedirs('results', exist_ok=True)
    with open('results/evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n=== RESULTS ===")
    for name, m in results.items():
        print(f"{name:15s}: R={m['reward_mean']:+.2f}±{m['reward_std']:.2f}  "
              f"SLA={m['sla_mean']:.1f}%  Thr={m['thr_mean']:.1f}Mbps  Lat={m['lat_mean']:.1f}ms")
    print("\nSaved to results/evaluation_results.json")
    env.close()


if __name__ == '__main__':
    main()
