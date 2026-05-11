# training/eval_baselines.py
#!/usr/bin/env python3
"""Baseline-only evaluation — run BEFORE DRL training to establish reference."""

import argparse, json, sys
from pathlib import Path
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.slice_gym_env import SliceGymEnv
from envs.metrics import compute_sla_rates
from training.evaluate import (
    evaluate_policy, random_policy,
    round_robin_policy, greedy_pf_policy,
    load_config
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",      type=int, default=5555)
    parser.add_argument("--episodes",  type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--config",    type=str, default="configs/config.yaml")
    parser.add_argument("--out",       type=str,
                        default="results/baseline_results.json")
    parser.add_argument("--seed",      type=int, default=99)
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    env = SliceGymEnv(port=args.port, sim_seed=args.seed, start_sim=False)
    print(f"[baselines] Connecting to NS-3 on port {args.port}...")
    obs, info = env.reset()
    print("[baselines] Connected.\n")

    results = {}
    rng = np.random.default_rng(args.seed)

    def _random(obs, hidden):
        return int(rng.integers(0, 27)), hidden

    obs, results["Random"]      = evaluate_policy(
        "Random",      _random,           env, obs,
        args.episodes, args.max_steps, cfg)
    obs, results["Round-Robin"] = evaluate_policy(
        "Round-Robin", round_robin_policy, env, obs,
        args.episodes, args.max_steps, cfg)
    obs, results["Greedy-PF"]   = evaluate_policy(
        "Greedy-PF",   greedy_pf_policy,  env, obs,
        args.episodes, args.max_steps, cfg)

    env.close()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\n[baselines] Results → {out}")

    # Print summary table
    print("\n" + "="*80)
    print(f"{'Policy':<14} {'Reward':>9} {'SLA%':>7} {'eMBB_Mbps':>10} "
          f"{'URLLC_ms':>9} {'ViolRate':>9}")
    print("-"*80)
    for name, r in results.items():
        print(f"{name:<14}"
              f" {r['mean_reward']:>9.3f}"
              f" {r['mean_sla_rate']*100:>6.1f}%"
              f" {r['mean_embb_thr_mbps']:>10.2f}"
              f" {r['mean_urllc_lat_ms']:>9.3f}"
              f" {r['mean_rwd_violation_rate']:>9.4f}")
    print("="*80)

if __name__ == "__main__":
    main()
