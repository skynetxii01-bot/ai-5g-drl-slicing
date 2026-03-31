import argparse
import json
from pathlib import Path

import numpy as np
import torch

from agents.dqn.dqn_agent import DQNAgent
from agents.ppo.ppo_agent import PPOAgent
from agents.r2d2.r2d2_agent import R2D2Agent
from envs.slice_gym_env import SliceGymEnv


def build_agent(name, device):
    if name == "dqn":
        return DQNAgent(device=device)
    if name == "ppo":
        return PPOAgent(device=device)
    if name == "r2d2":
        return R2D2Agent(device=device)
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", choices=["dqn", "ppo", "r2d2"], required=True)
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--episodes", type=int, default=500)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env = SliceGymEnv(port=args.port, seed=args.seed)
    agent = build_agent(args.agent, device="cpu")

    out_log = Path("results/logs")
    out_model = Path("results/models")
    out_log.mkdir(parents=True, exist_ok=True)
    out_model.mkdir(parents=True, exist_ok=True)
    log_path = out_log / f"{args.agent}_log.jsonl"

    for ep in range(1, args.episodes + 1):
        obs, info = env.reset(seed=args.seed + ep)
        total_r, steps = 0.0, 0
        done = False
        last_value = 0.0
        while not done:
            if args.agent == "ppo":
                action, logp, value = agent.act(obs)
                next_obs, reward, done, trunc, i2 = env.step(action)
                done = done or trunc
                agent.buf.add(obs, action, logp, reward, done, value)
                last_value = value
            else:
                action = agent.act(obs)
                next_obs, reward, done, trunc, i2 = env.step(action)
                done = done or trunc
                agent.update(obs, action, reward, next_obs, float(done))
            obs = next_obs
            total_r += reward
            steps += 1
        if args.agent == "ppo":
            agent.update(last_value=last_value)

        metrics = {
            "episode": ep,
            "reward": round(total_r, 4),
            "steps": steps,
            "embb_thr": round(float(i2["throughput"]["eMBB"] * 100), 4),
            "urllc_lat": round(float(i2["latency"]["URLLC"] * 100), 4),
            "sla_rate": round(float(np.mean(np.array(list(i2["latency"].values())) < 0.2)), 4),
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(metrics) + "\n")
        print(f"Episode {ep}/{args.episodes} | Reward: {total_r:.2f} | Steps: {steps} | eMBB: {metrics['embb_thr']:.2f} Mbps | URLLC: {metrics['urllc_lat']:.2f}ms | SLA: {metrics['sla_rate']*100:.1f}%")

        if ep % 100 == 0:
            model = agent.net.state_dict() if args.agent == "ppo" else agent.q.state_dict()
            torch.save(model, out_model / f"{args.agent}_ep{ep}.pt")

    model = agent.net.state_dict() if args.agent == "ppo" else agent.q.state_dict()
    torch.save(model, out_model / f"{args.agent}_final.pt")


if __name__ == "__main__":
    main()
