#!/usr/bin/env python3
"""Evaluate trained DRL policies and classical baselines against NS-3.

Architecture (CRITICAL — matches training):
  env.reset() is called ONCE before all evaluations begin.
  Each policy is then evaluated for --episodes consecutive step-count windows
  (--max-steps steps each) on the same continuous NS-3 simulation.

Baselines (all NS-3 connected):
  Random      — uniform random from 27 PRB-delta actions.
  Round-Robin — equal PRB allocation (target 9/8/8).
  Greedy-PF   — proportional fair heuristic, one-step lookahead.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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
from agents.r2d2.r2d2_network import R2D2Net
from envs.slice_gym_env import SliceGymEnv, SLICE_NAMES
from envs.metrics import compute_sla_rates, nan_or_round

PolicyFn = Callable[[np.ndarray, Any], Tuple[int, Any]]

_ACTION_NO_CHANGE = 13   # (0,0,0) delta — see _action_from_delta


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)




def _action_from_delta(d_embb: int, d_urllc: int, d_mmtc: int) -> int:
    return (d_embb + 1) * 9 + (d_urllc + 1) * 3 + (d_mmtc + 1)


# ---------------------------------------------------------------------------
# Baseline policies
# ---------------------------------------------------------------------------

def random_policy(obs: np.ndarray, hidden: Any) -> Tuple[int, Any]:
    return int(np.random.randint(0, 27)), hidden


def round_robin_policy(obs: np.ndarray, hidden: Any) -> Tuple[int, Any]:
    target_frac  = np.array([9.0 / 25, 8.0 / 25, 8.0 / 25])
    current_frac = np.array([obs[0], obs[1], obs[2]])
    deficit      = target_frac - current_frac
    donor        = int(np.argmin(deficit))
    receiver     = int(np.argmax(deficit))
    threshold    = 1.0 / 25
    if deficit[receiver] < threshold and abs(deficit[donor]) < threshold:
        return _ACTION_NO_CHANGE, hidden
    delta = [0, 0, 0]
    delta[receiver] =  1
    delta[donor]    = -1
    return _action_from_delta(delta[0], delta[1], delta[2]), hidden


def greedy_pf_policy(obs: np.ndarray, hidden: Any) -> Tuple[int, Any]:
    prb_frac   = np.array([obs[0], obs[1], obs[2]]) + 1e-9
    thr_norm   = np.array([obs[3], obs[4], obs[5]])
    efficiency = thr_norm / prb_frac
    active     = thr_norm > 1e-4
    if active.sum() < 2:
        return _ACTION_NO_CHANGE, hidden
    donor    = int(np.argmax(np.where(active, efficiency, -np.inf)))
    receiver = int(np.argmin(np.where(active, efficiency,  np.inf)))
    if donor == receiver:
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
    """Run `episodes` consecutive windows for one policy, return (obs, results)."""
    print(f"\n[evaluate.py] Evaluating {name} ({episodes} ep × {max_steps} steps)...")

    hidden = None

    # Per-episode collector lists
    ep_rewards:     List[float] = []
    ep_sla:         List[float] = []
    ep_sla_embb:    List[float] = []
    ep_sla_urllc:   List[float] = []
    ep_sla_mmtc:    List[float] = []
    ep_embb_thrs:   List[float] = []
    ep_urllc_thrs:  List[float] = []
    ep_mmtc_thrs:   List[float] = []
    ep_embb_lats:   List[float] = []
    ep_urllc_lats:  List[float] = []
    ep_mmtc_lats:   List[float] = []
    ep_hol_embb:    List[float] = []
    ep_hol_urllc:   List[float] = []
    ep_hol_mmtc:    List[float] = []
    ep_eff_embb:    List[float] = []
    ep_eff_urllc:   List[float] = []
    ep_eff_mmtc:    List[float] = []
    ep_prb_embb:    List[float] = []
    ep_prb_urllc:   List[float] = []
    ep_prb_mmtc:    List[float] = []
    ep_rwd_thr:     List[float] = []
    ep_rwd_sla:     List[float] = []
    ep_rwd_eff:     List[float] = []
    ep_rwd_viol:    List[float] = []
    ep_active:      List[float] = []

    for ep in range(episodes):
        # Step-level accumulators
        ep_reward        = 0.0
        sla_sum          = 0.0
        sla_embb_sum     = 0.0
        sla_urllc_sum    = 0.0
        sla_mmtc_sum     = 0.0
        embb_sum         = 0.0
        urllc_thr_sum    = 0.0
        mmtc_sum         = 0.0
        embb_lat_sum     = 0.0;  embb_lat_n  = 0
        urllc_lat_sum    = 0.0;  urllc_lat_n = 0
        mmtc_lat_sum     = 0.0;  mmtc_lat_n  = 0
        hol_embb_sum     = 0.0
        hol_urllc_sum    = 0.0
        hol_mmtc_sum     = 0.0
        eff_embb_sum     = 0.0
        eff_urllc_sum    = 0.0
        eff_mmtc_sum     = 0.0
        prb_embb_sum     = 0.0
        prb_urllc_sum    = 0.0
        prb_mmtc_sum     = 0.0
        rwd_thr_sum      = 0.0
        rwd_sla_sum      = 0.0
        rwd_eff_sum      = 0.0
        rwd_viol_sum     = 0.0
        active_sum       = 0.0
        step_count       = 0
        decoded: Dict    = {}

        for _ in range(max_steps):
            action, hidden = policy_fn(obs, hidden)
            obs, reward, done, truncated, info = env.step(action)
            decoded    = info.get("decoded_obs", decoded)
            ep_reward += float(reward)
            step_count += 1

            extra_json = info.get("extra_json")
            if extra_json is None and len(obs) >= 18:
                extra_json = {
                    "demand_active": [
                        int(round(float(obs[15]))),
                        int(round(float(obs[16]))),
                        int(round(float(obs[17]))),
                    ]
                }

            # SLA
            rates = compute_sla_rates(decoded, cfg, extra_json)
            sla_sum       += rates["overall"]
            sla_embb_sum  += rates["embb"]
            sla_urllc_sum += rates["urllc"]
            sla_mmtc_sum  += rates["mmtc"]

            # Throughput
            embb_sum      += float(decoded.get("throughput", {}).get("eMBB",  0.0)) * float(cfg["env"]["max_thr_mbps"]["eMBB"])
            urllc_thr_sum += float(decoded.get("throughput", {}).get("URLLC", 0.0)) * float(cfg["env"]["max_thr_mbps"]["URLLC"])
            mmtc_sum      += float(decoded.get("throughput", {}).get("mMTC",  0.0)) * float(cfg["env"]["max_thr_mbps"]["mMTC"])

            # Latency — prefer extra_json["lat_ms"], fall back to obs
            lat_ms_raw = extra_json.get("lat_ms") if extra_json else None

            def _accum_lat(slice_key, lat_idx, lat_sum, lat_n, max_lat_key):
                thr = (float(decoded.get("throughput", {}).get(slice_key, 0.0))
                       * float(cfg["env"]["max_thr_mbps"][slice_key]))
                if thr < 0.001:
                    return lat_sum, lat_n
                if lat_ms_raw is not None:
                    return lat_sum + float(lat_ms_raw[lat_idx]), lat_n + 1
                lat_norm = float(decoded.get("latency", {}).get(slice_key, 0.0))
                return lat_sum + lat_norm * 2.0 * float(cfg["env"]["max_lat_ms"][slice_key]), lat_n + 1

            embb_lat_sum,  embb_lat_n  = _accum_lat("eMBB",  0, embb_lat_sum,  embb_lat_n,  "eMBB")
            urllc_lat_sum, urllc_lat_n = _accum_lat("URLLC", 1, urllc_lat_sum, urllc_lat_n, "URLLC")
            mmtc_lat_sum,  mmtc_lat_n  = _accum_lat("mMTC",  2, mmtc_lat_sum,  mmtc_lat_n,  "mMTC")

            # HOL delay
            if extra_json and isinstance(extra_json.get("hol_norm"), list):
                hol_embb_sum  += float(extra_json["hol_norm"][0])
                hol_urllc_sum += float(extra_json["hol_norm"][1])
                hol_mmtc_sum  += float(extra_json["hol_norm"][2])
            elif len(obs) >= 12:
                hol_embb_sum  += float(obs[9])
                hol_urllc_sum += float(obs[10])
                hol_mmtc_sum  += float(obs[11])

            # PRB efficiency
            if len(obs) >= 15:
                eff_embb_sum  += float(obs[12])
                eff_urllc_sum += float(obs[13])
                eff_mmtc_sum  += float(obs[14])

            # PRB allocation
            prb_frac = decoded.get("prb_frac", {})
            prb_embb_sum  += prb_frac.get("eMBB",  0.4)  * 25
            prb_urllc_sum += prb_frac.get("URLLC", 0.32) * 25
            prb_mmtc_sum  += prb_frac.get("mMTC",  0.28) * 25

            # Reward decomposition
            rt = (extra_json or {}).get("reward_terms", {})
            if rt:
                rwd_thr_sum  += float(rt.get("thr_norm_avg",    0.0))
                rwd_sla_sum  += float(rt.get("sla_margin_norm", 0.0))
                rwd_eff_sum  += float(rt.get("eff_norm",        0.0))
                rwd_viol_sum += float(rt.get("violation_rate",  0.0))
                active_sum   += float(rt.get("active_slices",   0.0))

            if done or truncated:
                break

        n = max(1, step_count)

        ep_rewards.append(ep_reward)
        ep_sla.append(sla_sum / n)
        ep_sla_embb.append(sla_embb_sum  / n)
        ep_sla_urllc.append(sla_urllc_sum / n)
        ep_sla_mmtc.append(sla_mmtc_sum  / n)
        ep_embb_thrs.append(embb_sum      / n)
        ep_urllc_thrs.append(urllc_thr_sum / n)
        ep_mmtc_thrs.append(mmtc_sum      / n)
        ep_embb_lats.append((embb_lat_sum  / embb_lat_n)  if embb_lat_n  > 0 else float("nan"))
        ep_urllc_lats.append((urllc_lat_sum / urllc_lat_n) if urllc_lat_n > 0 else float("nan"))
        ep_mmtc_lats.append((mmtc_lat_sum  / mmtc_lat_n)  if mmtc_lat_n  > 0 else float("nan"))
        ep_hol_embb.append(hol_embb_sum  / n)
        ep_hol_urllc.append(hol_urllc_sum / n)
        ep_hol_mmtc.append(hol_mmtc_sum  / n)
        ep_eff_embb.append(eff_embb_sum  / n)
        ep_eff_urllc.append(eff_urllc_sum / n)
        ep_eff_mmtc.append(eff_mmtc_sum  / n)
        ep_prb_embb.append(prb_embb_sum  / n)
        ep_prb_urllc.append(prb_urllc_sum / n)
        ep_prb_mmtc.append(prb_mmtc_sum  / n)
        ep_rwd_thr.append(rwd_thr_sum  / n)
        ep_rwd_sla.append(rwd_sla_sum  / n)
        ep_rwd_eff.append(rwd_eff_sum  / n)
        ep_rwd_viol.append(rwd_viol_sum / n)
        ep_active.append(active_sum   / n)

        print(f"  ep {ep+1:>2}/{episodes}  "
              f"reward={ep_reward:>8.3f}  "
              f"sla={sla_sum/n:.3f}  "
              f"eMBB={embb_sum/n:.1f}Mbps  "
              f"URLLC_lat={urllc_lat_sum/max(1,urllc_lat_n):.1f}ms  "
              f"viol={rwd_viol_sum/n:.3f}")

    def _mean(lst):  return float(np.nanmean(lst))
    def _std(lst):   return float(np.nanstd(lst))

    results = {
        # reward
        "mean_reward":            _mean(ep_rewards),
        "std_reward":             _std(ep_rewards),
        # SLA — overall
        "mean_sla_rate":          _mean(ep_sla),
        "std_sla_rate":           _std(ep_sla),
        # SLA — per slice
        "mean_sla_embb":          _mean(ep_sla_embb),
        "std_sla_embb":           _std(ep_sla_embb),
        "mean_sla_urllc":         _mean(ep_sla_urllc),
        "std_sla_urllc":          _std(ep_sla_urllc),
        "mean_sla_mmtc":          _mean(ep_sla_mmtc),
        "std_sla_mmtc":           _std(ep_sla_mmtc),
        # throughput (Mbps)
        "mean_embb_thr_mbps":     _mean(ep_embb_thrs),
        "std_embb_thr_mbps":      _std(ep_embb_thrs),
        "mean_urllc_thr_mbps":    _mean(ep_urllc_thrs),
        "std_urllc_thr_mbps":     _std(ep_urllc_thrs),
        "mean_mmtc_thr_mbps":     _mean(ep_mmtc_thrs),
        "std_mmtc_thr_mbps":      _std(ep_mmtc_thrs),
        # latency (ms)
        "mean_embb_lat_ms":       _mean(ep_embb_lats),
        "std_embb_lat_ms":        _std(ep_embb_lats),
        "mean_urllc_lat_ms":      _mean(ep_urllc_lats),
        "std_urllc_lat_ms":       _std(ep_urllc_lats),
        "mean_mmtc_lat_ms":       _mean(ep_mmtc_lats),
        "std_mmtc_lat_ms":        _std(ep_mmtc_lats),
        # HOL delay (normalised)
        "mean_hol_embb":          _mean(ep_hol_embb),
        "mean_hol_urllc":         _mean(ep_hol_urllc),
        "mean_hol_mmtc":          _mean(ep_hol_mmtc),
        # PRB efficiency (normalised)
        "mean_eff_embb":          _mean(ep_eff_embb),
        "mean_eff_urllc":         _mean(ep_eff_urllc),
        "mean_eff_mmtc":          _mean(ep_eff_mmtc),
        # PRB allocation
        "mean_prb_embb":          _mean(ep_prb_embb),
        "std_prb_embb":           _std(ep_prb_embb),
        "mean_prb_urllc":         _mean(ep_prb_urllc),
        "std_prb_urllc":          _std(ep_prb_urllc),
        "mean_prb_mmtc":          _mean(ep_prb_mmtc),
        "std_prb_mmtc":           _std(ep_prb_mmtc),
        # reward decomposition
        "mean_rwd_thr_norm":      _mean(ep_rwd_thr),
        "mean_rwd_sla_margin":    _mean(ep_rwd_sla),
        "mean_rwd_eff_norm":      _mean(ep_rwd_eff),
        "mean_rwd_violation_rate":_mean(ep_rwd_viol),
        "mean_active_slices":     _mean(ep_active),
        # meta
        "episodes": episodes,
    }
    return obs, results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",      type=int, default=5555)
    parser.add_argument("--episodes",  type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--config",    type=str, default="configs/config.yaml")
    parser.add_argument("--out",       type=str, default="results/evaluation_results.json")
    parser.add_argument("--seed",      type=int, default=99)
    parser.add_argument("--device",    type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg       = load_config(Path(args.config))
    device    = torch.device(args.device)
    model_dir = PROJECT_ROOT / "results" / "models"

    print(f"[evaluate.py] Device:   {device}")
    print(f"[evaluate.py] Episodes: {args.episodes} per policy")

    env = SliceGymEnv(port=args.port, sim_seed=args.seed, start_sim=False)
    print(f"[evaluate.py] Connecting to NS-3 on port {args.port}...")
    obs, info = env.reset()
    print("[evaluate.py] Connected.\n")

    results: Dict[str, Dict] = {}

    # ── DQN ──────────────────────────────────────────────────────────────────
    dqn_path = model_dir / "dqn_final.pt"
    if dqn_path.exists():
        dqn  = DuelingDQN(18, 27).to(device)
        ckpt = torch.load(dqn_path, map_location=device)
        dqn.load_state_dict(ckpt["online"])
        dqn.eval()

        def dqn_policy(obs, hidden):
            with torch.no_grad():
                t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                q = dqn(t)
            return int(torch.argmax(q, dim=1).item()), hidden

        obs, results["DQN"] = evaluate_policy(
            "DQN", dqn_policy, env, obs, args.episodes, args.max_steps, cfg)
    else:
        print(f"[evaluate.py] SKIP DQN — {dqn_path} not found")

    # ── PPO ──────────────────────────────────────────────────────────────────
    ppo_path = model_dir / "ppo_final.pt"
    if ppo_path.exists():
        ppo  = ActorCritic(18, 27).to(device)
        ckpt = torch.load(ppo_path, map_location=device)
        ppo.load_state_dict(ckpt.get("model", ckpt))
        ppo.eval()

        def ppo_policy(obs, hidden):
            with torch.no_grad():
                t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                logits, _ = ppo(t)
                action = int(Categorical(logits=logits).probs.argmax(dim=-1).item())
            return action, hidden

        obs, results["PPO"] = evaluate_policy(
            "PPO", ppo_policy, env, obs, args.episodes, args.max_steps, cfg)
    else:
        print(f"[evaluate.py] SKIP PPO — {ppo_path} not found")

    # ── R2D2 ─────────────────────────────────────────────────────────────────
    r2d2_path = model_dir / "r2d2_final.pt"
    if r2d2_path.exists():
        r2d2 = R2D2Net(18, 27).to(device)
        ckpt = torch.load(r2d2_path, map_location=device)
        r2d2.load_state_dict(ckpt["online"])
        r2d2.eval()

        def r2d2_policy(obs, hidden):
            with torch.no_grad():
                t = torch.tensor(obs, dtype=torch.float32, device=device).view(1, 1, -1)
                q, hidden_out = r2d2(t, hidden)
            return int(torch.argmax(q[0, -1]).item()), hidden_out

        obs, results["R2D2"] = evaluate_policy(
            "R2D2", r2d2_policy, env, obs, args.episodes, args.max_steps, cfg)
    else:
        print(f"[evaluate.py] SKIP R2D2 — {r2d2_path} not found")

    # ── Baselines ─────────────────────────────────────────────────────────────
    rng = np.random.default_rng(args.seed)

    def _random_policy(obs, hidden):
        return int(rng.integers(0, 27)), hidden

    obs, results["Random"]       = evaluate_policy(
        "Random",      _random_policy,    env, obs, args.episodes, args.max_steps, cfg)
    obs, results["Round-Robin"]  = evaluate_policy(
        "Round-Robin", round_robin_policy, env, obs, args.episodes, args.max_steps, cfg)
    obs, results["Greedy-PF"]    = evaluate_policy(
        "Greedy-PF",   greedy_pf_policy,  env, obs, args.episodes, args.max_steps, cfg)

    env.close()

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[evaluate.py] Results → {out_path}")

    # ── Summary table ─────────────────────────────────────────────────────────
    col = 14
    print("\n" + "=" * 100)
    print(f"{'Policy':<{col}} {'Reward':>9} {'SLA%':>7} "
          f"{'eMBB_SLA':>9} {'URLLC_SLA':>10} {'mMTC_SLA':>9} "
          f"{'eMBB_Mbps':>10} {'URLLC_ms':>9} {'ViolRate':>9}")
    print("-" * 100)
    for name, r in results.items():
        print(
            f"{name:<{col}}"
            f" {r['mean_reward']:>9.3f}"
            f" {r['mean_sla_rate']*100:>6.1f}%"
            f" {r['mean_sla_embb']*100:>8.1f}%"
            f" {r['mean_sla_urllc']*100:>9.1f}%"
            f" {r['mean_sla_mmtc']*100:>8.1f}%"
            f" {r['mean_embb_thr_mbps']:>10.2f}"
            f" {r['mean_urllc_lat_ms']:>9.3f}"
            f" {r['mean_rwd_violation_rate']:>9.4f}"
        )
    print("=" * 100)


if __name__ == "__main__":
    main()
