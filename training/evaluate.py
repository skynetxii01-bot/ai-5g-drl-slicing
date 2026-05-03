#!/usr/bin/env python3
"""Evaluate trained DRL policies and classical baselines against NS-3.

Architecture (CRITICAL — matches training):
  env.reset() is called ONCE before all evaluations begin.
  Each policy is then evaluated for --episodes consecutive step-count windows
  (--max-steps steps each) on the same continuous NS-3 simulation.
  This is identical to the training architecture and is required by the
  ns3-gym ZMQ protocol — NS-3 cannot truly reset mid-simulation.

  Policies are evaluated sequentially. The simulation state at the end of
  policy N is the starting state for policy N+1. Over many steps, the
  exponential OnOff traffic process is ergodic, so each policy sees a
  statistically representative sample of the traffic workload.

Wall-clock time warning:
  Each episode ≈ 41.6 minutes. With --episodes 3 and 6 policies (3 DRL +
  3 baselines), total evaluation time ≈ 18 hours. Plan accordingly.
  Run with --episodes 1 for a quick sanity check.

Baselines (all NS-3 connected):
  Random:        Uniform random from the 27 PRB-delta actions each step.
                 Establishes a lower bound — any trained policy should exceed this.

  Round-Robin:   Maintains equal PRB allocation (target: 9/8/8 across eMBB/URLLC/mMTC).
                 At each step, shifts one PRB from the slice most above its equal share
                 to the slice most below. Represents a demand-agnostic static allocation
                 policy — the classical baseline in slicing literature.

  Greedy-PF:     Proportional Fair-inspired heuristic. At each step, computes
                 throughput per unit PRB for each active slice. Shifts one PRB from
                 the most efficient slice (has headroom) to the least efficient
                 (most underserved relative to its allocation). This is a one-step
                 lookahead policy with no memory — a strong non-learning baseline.

Usage:
  # Launch NS-3 first (separate terminal), then:
  python training/evaluate.py --port 5555 --episodes 3 --max-steps 1000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from torch.distributions import Categorical

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.dqn.dqn_network import DuelingDQN
from agents.ppo.actor_critic import ActorCritic
from agents.r2d2.r2d2_network import R2D2Net          # FIX: was R2D2Network (wrong class name)
from envs.slice_gym_env import SliceGymEnv, SLICE_NAMES

# ---------------------------------------------------------------------------
# Type alias: a policy function takes (obs, hidden) and returns (action, hidden).
# DQN and baselines pass hidden through unchanged. R2D2 uses hidden as LSTM state.
# ---------------------------------------------------------------------------
PolicyFn = Callable[[np.ndarray, Any], Tuple[int, Any]]

# Action index for zero delta (hold current allocation).
# Formula: (delta_embb+1)*9 + (delta_urllc+1)*3 + (delta_mmtc+1)
# With all deltas = 0: (0+1)*9 + (0+1)*3 + (0+1) = 13.
_ACTION_NO_CHANGE = 13


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_sla_rate(decoded_obs: Dict, cfg: Dict) -> float:
    """Fraction of slices meeting SLA this step.

    Must stay identical to train.py:compute_sla_rate. Both functions must agree
    so that evaluation metrics are directly comparable to training metrics.

    Off-period exclusion: mMTC and URLLC are counted as satisfied when their
    observed throughput is below 0.001 Mbps (on/off traffic model off-period).
    Latency boundary: obs[6:9] is normalised by 2*maxLatMs, so the SLA
    threshold sits at lat_norm=0.5, not 1.0.
    """
    max_thr = cfg["env"]["max_thr_mbps"]
    min_thr = cfg["env"]["min_thr_mbps"]
    sat = 0
    for s in SLICE_NAMES:
        thr      = float(decoded_obs["throughput"][s]) * float(max_thr[s])
        lat_norm = float(decoded_obs["latency"][s])

        mmtc_inactive  = (s == "mMTC"  and thr < 0.001)
        urllc_inactive = (s == "URLLC" and thr < 0.001)
        slice_inactive = mmtc_inactive or urllc_inactive

        if slice_inactive or (thr >= float(min_thr[s]) and lat_norm < 0.5):
            sat += 1
    return sat / len(SLICE_NAMES)


def _action_from_delta(d_embb: int, d_urllc: int, d_mmtc: int) -> int:
    """Encode a (eMBB-delta, URLLC-delta, mMTC-delta) triple as an action index.

    Each delta must be in {-1, 0, +1}. The encoding matches kActionDelta in
    slice-env.cc — action = (d+1)*9 + (d+1)*3 + (d+1) for eMBB, URLLC, mMTC.

    Verification spot-checks:
      (0, 0, 0) -> 13 (no change)     [action 13 in kActionDelta]
      (1, 0, 0) -> 22 (eMBB +1 only)  [action 22: {1,0,0}]
      (-1,0, 1) ->  2 (eMBB-1,mMTC+1) [action 2: {-1,-1,1}... wait let me recheck]

    Actually from the table:
      action 2: {-1,-1,1} = (-1,-1,1)
      (-1+1)*9 + (-1+1)*3 + (1+1) = 0 + 0 + 2 = 2 ✓
    """
    return (d_embb + 1) * 9 + (d_urllc + 1) * 3 + (d_mmtc + 1)


# ---------------------------------------------------------------------------
# Baseline policies
# All have signature: (obs: np.ndarray, hidden: Any) -> (action: int, hidden: Any)
# ---------------------------------------------------------------------------

def random_policy(obs: np.ndarray, hidden: Any) -> Tuple[int, Any]:
    """Uniform random selection from all 27 PRB-delta actions."""
    return int(np.random.randint(0, 27)), hidden


def round_robin_policy(obs: np.ndarray, hidden: Any) -> Tuple[int, Any]:
    """Equal-allocation policy: drive PRBs toward the fair share of 9/8/8.

    At each step, identifies which slice is most above its fair share (donor)
    and which is most below (receiver), then shifts one PRB accordingly.
    When all slices are within 1 PRB of their fair share, takes no action.

    This is the classical Round-Robin interpretation for PRB allocation:
    treat all slices equally regardless of traffic demand. In network slicing
    literature this is the baseline that DRL must decisively outperform.

    Target allocation: [9, 8, 8] / 25 = [0.36, 0.32, 0.32]
    (closest integer partition of 25 into 3 equal parts)
    """
    target_frac = np.array([9.0 / 25, 8.0 / 25, 8.0 / 25])
    current_frac = np.array([obs[0], obs[1], obs[2]])
    deficit = target_frac - current_frac   # positive = needs more, negative = has too much

    donor    = int(np.argmin(deficit))     # most above target → gives -1
    receiver = int(np.argmax(deficit))     # most below target → gets +1

    # If already close enough to equal allocation, hold steady.
    threshold = 1.0 / 25   # = 1 PRB fraction
    if deficit[receiver] < threshold and abs(deficit[donor]) < threshold:
        return _ACTION_NO_CHANGE, hidden

    delta = [0, 0, 0]
    delta[receiver] =  1
    delta[donor]    = -1
    return _action_from_delta(delta[0], delta[1], delta[2]), hidden


def greedy_pf_policy(obs: np.ndarray, hidden: Any) -> Tuple[int, Any]:
    """Proportional Fair-inspired greedy heuristic.

    Computes throughput efficiency (normalised throughput / PRB fraction) for
    each active slice. Shifts one PRB from the most efficient slice (has
    headroom — high throughput per PRB) to the least efficient slice (most
    underserved — low throughput per PRB). This mirrors the PF scheduler's
    principle of allocating resources where marginal gain is highest relative
    to current allocation.

    Inactive slices (off-period throughput < 1e-4 normalised) are excluded
    from both donor and receiver selection — they should neither give away
    their PRBs (they may need them when they wake up) nor receive extra ones
    (no traffic to use them).

    Requires at least 2 active slices to perform a meaningful swap.
    """
    prb_frac = np.array([obs[0], obs[1], obs[2]]) + 1e-9  # avoid division by zero
    thr_norm  = np.array([obs[3], obs[4], obs[5]])
    efficiency = thr_norm / prb_frac   # throughput per unit PRB for each slice

    # An active slice has meaningful throughput in the current 100ms window.
    # Threshold 1e-4 (normalised) corresponds to roughly 0.001–0.01 Mbps
    # depending on the slice's maxThrMbps — safely above noise floor.
    active = thr_norm > 1e-4

    if active.sum() < 2:
        # Cannot perform a meaningful swap with fewer than 2 active slices.
        return _ACTION_NO_CHANGE, hidden

    # Donor: highest efficiency among active slices (can afford to give up a PRB)
    eff_for_donor = np.where(active, efficiency, -np.inf)
    donor = int(np.argmax(eff_for_donor))

    # Receiver: lowest efficiency among active slices (most underserved per PRB)
    eff_for_receiver = np.where(active, efficiency, np.inf)
    receiver = int(np.argmin(eff_for_receiver))

    if donor == receiver:
        # All active slices have identical efficiency — no swap is beneficial.
        return _ACTION_NO_CHANGE, hidden

    delta = [0, 0, 0]
    delta[donor]    = -1
    delta[receiver] =  1
    return _action_from_delta(delta[0], delta[1], delta[2]), hidden


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------

def evaluate_policy(
    name: str,
    policy_fn: PolicyFn,
    env: SliceGymEnv,
    obs: np.ndarray,
    episodes: int,
    max_steps: int,
    cfg: Dict,
) -> Tuple[np.ndarray, Dict]:
    """Run `episodes` consecutive step-count evaluation windows for one policy.

    Returns the updated observation (to seed the next policy evaluation) and
    a dict of aggregated metrics over all episodes.

    Hidden state is reset to None at the start of this policy's evaluation.
    For DQN and baselines this has no effect. For R2D2 it clears the LSTM
    state — each policy starts with a fresh recurrent context.
    """
    print(f"\n[evaluate.py] Evaluating {name} ({episodes} episodes × {max_steps} steps)...")

    hidden = None   # reset recurrent state between policies
    ep_rewards:    List[float] = []
    ep_sla_rates:  List[float] = []
    ep_embb_thrs:  List[float] = []
    ep_urllc_lats: List[float] = []
    ep_prb_embb:   List[float] = []
    ep_prb_urllc:  List[float] = []
    ep_prb_mmtc:   List[float] = []

    for ep in range(episodes):
        ep_reward    = 0.0
        sla_sum      = 0.0
        embb_sum     = 0.0
        urllc_lat_sum = 0.0
        prb_embb_sum  = 0.0
        prb_urllc_sum = 0.0
        prb_mmtc_sum  = 0.0
        step_count    = 0
        decoded: Dict = {}

        for _ in range(max_steps):
            action, hidden = policy_fn(obs, hidden)
            obs, reward, done, truncated, info = env.step(action)
            decoded    = info.get("decoded_obs", decoded)
            ep_reward += float(reward)
            step_count += 1

            # Accumulate step-level metrics for episode averages.
            sla_sum       += compute_sla_rate(decoded, cfg)
            embb_sum      += float(decoded.get("throughput", {}).get("eMBB",  0.0)) \
                             * float(cfg["env"]["max_thr_mbps"]["eMBB"])
            urllc_lat_sum += float(decoded.get("latency",    {}).get("URLLC", 0.0)) \
                             * 2.0 * float(cfg["env"]["max_lat_ms"]["URLLC"])
            prb_frac       = decoded.get("prb_frac", {})
            prb_embb_sum  += prb_frac.get("eMBB",  0.4)  * 25
            prb_urllc_sum += prb_frac.get("URLLC", 0.32) * 25
            prb_mmtc_sum  += prb_frac.get("mMTC",  0.28) * 25

            if done or truncated:
                break

        n = max(1, step_count)
        ep_rewards.append(ep_reward)
        ep_sla_rates.append(sla_sum       / n)
        ep_embb_thrs.append(embb_sum      / n)
        ep_urllc_lats.append(urllc_lat_sum / n)
        ep_prb_embb.append(prb_embb_sum   / n)
        ep_prb_urllc.append(prb_urllc_sum / n)
        ep_prb_mmtc.append(prb_mmtc_sum   / n)

        print(f"  ep {ep + 1:>2}/{episodes}  "
              f"reward={ep_reward:>8.3f}  "
              f"sla={sla_sum/n:.2f}  "
              f"eMBB={embb_sum/n:.1f}Mbps  "
              f"URLLC_lat={urllc_lat_sum/n:.2f}ms")

    results = {
        "mean_reward":       float(np.mean(ep_rewards)),
        "std_reward":        float(np.std(ep_rewards)),
        "mean_sla_rate":     float(np.mean(ep_sla_rates)),
        "std_sla_rate":      float(np.std(ep_sla_rates)),
        "mean_embb_thr_mbps": float(np.mean(ep_embb_thrs)),
        "std_embb_thr_mbps":  float(np.std(ep_embb_thrs)),
        "mean_urllc_lat_ms":  float(np.mean(ep_urllc_lats)),
        "std_urllc_lat_ms":   float(np.std(ep_urllc_lats)),
        "mean_prb_embb":      float(np.mean(ep_prb_embb)),
        "mean_prb_urllc":     float(np.mean(ep_prb_urllc)),
        "mean_prb_mmtc":      float(np.mean(ep_prb_mmtc)),
        "episodes":           episodes,
    }
    return obs, results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DRL policies and baselines in NS-3.")
    parser.add_argument("--port",      type=int,  default=5555)
    parser.add_argument("--episodes",  type=int,  default=3,
                        help="Episodes per policy. Default 3 ≈ 18h wall-clock. "
                             "Use 1 for a quick sanity check.")
    parser.add_argument("--max-steps", type=int,  default=1000)
    parser.add_argument("--config",    type=str,  default="configs/config.yaml")
    parser.add_argument("--out",       type=str,  default="results/evaluation_results.json")
    parser.add_argument("--seed",      type=int,  default=99)
    parser.add_argument("--device",    type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg       = load_config(Path(args.config))
    device    = torch.device(args.device)
    model_dir = PROJECT_ROOT / "results" / "models"

    print(f"[evaluate.py] Device: {device}")
    print(f"[evaluate.py] Episodes per policy: {args.episodes}")
    print(f"[evaluate.py] Estimated wall-clock: "
          f"~{args.episodes * 6 * 41.6:.0f} min total\n")

    # ── Connect to NS-3 — SINGLE reset, never repeated ────────────────────────
    # env.reset() is called here and NOWHERE ELSE in this file.
    # All policies share the same continuous NS-3 simulation.
    env = SliceGymEnv(port=args.port, sim_seed=args.seed, start_sim=False)
    print(f"[evaluate.py] Connecting to NS-3 on port {args.port}...")
    obs, info = env.reset()
    print("[evaluate.py] Connected. Beginning evaluation.\n")

    results: Dict[str, Dict] = {}

    # ── DRL agent: DQN ────────────────────────────────────────────────────────
    dqn_path = model_dir / "dqn_final.pt"
    if dqn_path.exists():
        dqn = DuelingDQN(15, 27).to(device)
        ckpt = torch.load(dqn_path, map_location=device)
        dqn.load_state_dict(ckpt["online"])
        dqn.eval()

        def dqn_policy(obs: np.ndarray, hidden: Any) -> Tuple[int, Any]:
            with torch.no_grad():
                t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                q = dqn(t)
            return int(torch.argmax(q, dim=1).item()), hidden  # greedy, hidden unused

        obs, results["DQN"] = evaluate_policy(
            "DQN", dqn_policy, env, obs, args.episodes, args.max_steps, cfg
        )
    else:
        print(f"[evaluate.py] SKIP DQN — checkpoint not found at {dqn_path}")

    # ── DRL agent: PPO ────────────────────────────────────────────────────────
    ppo_path = model_dir / "ppo_final.pt"
    if ppo_path.exists():
        ppo = ActorCritic(15, 27).to(device)
        ckpt = torch.load(ppo_path, map_location=device)
        # Checkpoint key is "model" (set by PpoAgent.save()) — fall back to raw dict.
        ppo.load_state_dict(ckpt.get("model", ckpt))
        ppo.eval()

        def ppo_policy(obs: np.ndarray, hidden: Any) -> Tuple[int, Any]:
            # FIX: was ppo.get_dist_value() which does not exist on ActorCritic.
            # ActorCritic.forward() returns (logits, value). We wrap logits in
            # Categorical and take the mode (argmax) for deterministic greedy eval.
            with torch.no_grad():
                t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                logits, _ = ppo(t)   # (1, 27) logits, (1, 1) value — we only need logits
                action = int(Categorical(logits=logits).probs.argmax(dim=-1).item())
            return action, hidden   # hidden unused for PPO

        obs, results["PPO"] = evaluate_policy(
            "PPO", ppo_policy, env, obs, args.episodes, args.max_steps, cfg
        )
    else:
        print(f"[evaluate.py] SKIP PPO — checkpoint not found at {ppo_path}")

    # ── DRL agent: R2D2 ───────────────────────────────────────────────────────
    r2d2_path = model_dir / "r2d2_final.pt"
    if r2d2_path.exists():
        # FIX: was R2D2Network — the actual class name is R2D2Net.
        r2d2 = R2D2Net(15, 27).to(device)
        ckpt = torch.load(r2d2_path, map_location=device)
        r2d2.load_state_dict(ckpt["online"])
        r2d2.eval()

        def r2d2_policy(obs: np.ndarray, hidden: Any) -> Tuple[int, Any]:
            # R2D2Net.forward() signature: (x: Tensor[batch, seq, obs_dim], hc) -> (q, hc)
            # For single-step inference: batch=1, seq=1.
            with torch.no_grad():
                t = torch.tensor(obs, dtype=torch.float32, device=device).view(1, 1, -1)
                q, hidden_out = r2d2(t, hidden)
            # q shape: (1, 1, 27) — take the last time-step's Q-values.
            return int(torch.argmax(q[0, -1]).item()), hidden_out

        obs, results["R2D2"] = evaluate_policy(
            "R2D2", r2d2_policy, env, obs, args.episodes, args.max_steps, cfg
        )
    else:
        print(f"[evaluate.py] SKIP R2D2 — checkpoint not found at {r2d2_path}")

    # ── Baselines (NS-3 connected — NOT synthetic) ────────────────────────────
    # Each baseline uses the real env.step() and receives real rewards from NS-3.
    # Results are directly comparable to DRL results above.
    rng = np.random.default_rng(args.seed)

    def _random_policy(obs: np.ndarray, hidden: Any) -> Tuple[int, Any]:
        return int(rng.integers(0, 27)), hidden

    obs, results["Random"]      = evaluate_policy(
        "Random",     _random_policy,    env, obs, args.episodes, args.max_steps, cfg
    )
    obs, results["Round-Robin"] = evaluate_policy(
        "Round-Robin", round_robin_policy, env, obs, args.episodes, args.max_steps, cfg
    )
    obs, results["Greedy-PF"]   = evaluate_policy(
        "Greedy-PF",  greedy_pf_policy,  env, obs, args.episodes, args.max_steps, cfg
    )

    env.close()

    # ── Save results ──────────────────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[evaluate.py] Results saved to {out_path}")

    # ── Print summary table ───────────────────────────────────────────────────
    col = 20
    print("\n" + "=" * 78)
    print(f"{'Policy':<{col}} {'Reward':>10} {'±':>5} {'SLA%':>7} {'eMBB(Mbps)':>12} {'URLLC(ms)':>11}")
    print("-" * 78)
    for name, r in results.items():
        print(
            f"{name:<{col}}"
            f" {r['mean_reward']:>10.3f}"
            f" {r['std_reward']:>5.2f}"
            f" {r['mean_sla_rate'] * 100:>6.1f}%"
            f" {r['mean_embb_thr_mbps']:>12.2f}"
            f" {r['mean_urllc_lat_ms']:>10.3f}"
        )
    print("=" * 78)


if __name__ == "__main__":
    main()
