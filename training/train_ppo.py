#!/usr/bin/env python3
"""Training entrypoint — PPO.

ns3-gym single-reset architecture — env.reset() called once before the
episode loop. Episodes are 1000-step windows of a continuous NS-3 run.
See train.py for full architecture notes.

PPO update schedule:
  buffer accumulates every env.step(). When len(buffer) >= n_steps OR
  the episode ends, update() is called. With n_steps=1000 == max_steps,
  one update fires at the end of every episode (standard on-policy rollout).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import deque
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
MODEL_DIR = PROJECT_ROOT / "results" / "models"

from agents.ppo.ppo_agent import PpoAgent, PpoConfig
from envs.slice_gym_env import SLICE_NAMES, SliceGymEnv
from envs.metrics import compute_sla_rates, nan_or_round
from monitor import TrainingMonitor


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_latest_checkpoint(model_dir: Path) -> Path | None:
    checkpoints = sorted(
        model_dir.glob("ppo_ep*.pt"),
        key=lambda p: int(p.stem.replace("ppo_ep", ""))
    )
    final = model_dir / "ppo_final.pt"
    if final.exists():
        return final
    return checkpoints[-1] if checkpoints else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",     type=int, default=5555)
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--config",   type=str, default="configs/config.yaml")
    parser.add_argument("--resume",   action="store_true")
    parser.add_argument("--device",   type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    if args.episodes is None:
        args.episodes = int(cfg["train"]["episodes"])

    set_seed(args.seed)
    device = torch.device(args.device)
    print(f"[train_ppo.py] Device: {device}")
    if device.type == "cuda":
        print(f"[train_ppo.py] GPU: {torch.cuda.get_device_name(0)}")

    env = SliceGymEnv(port=args.port, sim_seed=args.seed, start_sim=False)

    ppo_raw = cfg.get("ppo", {})
    ppo_cfg = PpoConfig(
        obs_dim    = 18,
        action_dim = 27,
        lr         = float(ppo_raw.get("lr",         3e-4)),
        gamma      = float(ppo_raw.get("gamma",      0.99)),
        lambda_gae = float(ppo_raw.get("lambda_gae", 0.95)),
        eps_clip   = float(ppo_raw.get("eps_clip",   0.2)),
        n_steps    = int(  ppo_raw.get("n_steps",    1000)),  # == max_steps → update every episode
        epochs     = int(  ppo_raw.get("epochs",     4)),
        batch_size = int(  ppo_raw.get("batch_size", 64)),
        ent_coef   = float(ppo_raw.get("ent_coef",   0.01)),
        vf_coef    = float(ppo_raw.get("vf_coef",    0.5)),
        grad_clip  = float(ppo_raw.get("grad_clip",  0.5)),
    )
    agent    = PpoAgent(ppo_cfg, device)
    start_ep = 1

    # ── Resume ────────────────────────────────────────────────────────────────
    model_dir = MODEL_DIR
    os.makedirs(model_dir, exist_ok=True)

    if args.resume:
        ckpt_path = find_latest_checkpoint(model_dir)
        if ckpt_path is not None:
            print(f"[train_ppo.py] Resuming from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device)
            agent.net.load_state_dict(ckpt["model"])
            agent.optim.load_state_dict(ckpt["optim"])
            log_path = PROJECT_ROOT / "results" / "logs" / "ppo_log.jsonl"
            if log_path.exists():
                with log_path.open() as f:
                    last = None
                    for line in f:
                        line = line.strip()
                        if line:
                            last = line
                try:
                    start_ep = (json.loads(last)["episode"] + 1) if last else 1
                except (json.JSONDecodeError, KeyError, TypeError):
                    start_ep = 1
            print(f"[train_ppo.py] Continuing from episode {start_ep}")
        else:
            print("[train_ppo.py] No checkpoint found — starting from scratch.")

    log_path = PROJECT_ROOT / "results" / "logs" / "ppo_log.jsonl"
    os.makedirs(log_path.parent, exist_ok=True)

    max_steps  = int(cfg["train"]["max_steps"])
    save_every = int(cfg["train"]["save_every"])

    print(f"[train_ppo.py] Connecting to NS-3 on port {args.port}...")
    obs, info = env.reset()
    print("[train_ppo.py] Connected. Starting training.")
    print(f"[train_ppo.py] obs[12:15] prb_efficiency = {obs[12]:.3f}, {obs[13]:.3f}, {obs[14]:.3f}")
    print(f"[train_ppo.py] obs[15:18] demand_active   = {obs[15]:.0f}, {obs[16]:.0f}, {obs[17]:.0f}")

    decoded = info.get("decoded_obs", None)
    reward_history: deque[float] = deque(maxlen=10)

    with TrainingMonitor("ppo", start_ep + args.episodes - 1, max_steps,
                         PROJECT_ROOT / "results") as mon:
        with log_path.open("a", encoding="utf-8") as logf:
            for ep in range(start_ep, start_ep + args.episodes):

                ep_reward  = 0.0
                step_count = 0
                sim_done   = False
                agent.buffer.clear()
                step_bar   = mon.begin_episode(ep)
                ep_losses: list[float] = []

                # ── Per-episode accumulators (mirrors train.py exactly) ────
                ep_embb_sum      = 0.0
                ep_urllc_thr_sum = 0.0
                ep_mmtc_sum      = 0.0
                ep_embb_lat_sum  = 0.0;  ep_embb_lat_n  = 0
                ep_urllc_lat_sum = 0.0;  ep_urllc_lat_n = 0
                ep_mmtc_lat_sum  = 0.0;  ep_mmtc_lat_n  = 0
                ep_sla_sum       = 0.0
                ep_sla_embb_sum  = 0.0
                ep_sla_urllc_sum = 0.0
                ep_sla_mmtc_sum  = 0.0
                ep_hol_embb_sum  = 0.0
                ep_hol_urllc_sum = 0.0
                ep_hol_mmtc_sum  = 0.0
                ep_eff_embb_sum  = 0.0
                ep_eff_urllc_sum = 0.0
                ep_eff_mmtc_sum  = 0.0
                ep_rwd_thr_sum   = 0.0
                ep_rwd_sla_sum   = 0.0
                ep_rwd_eff_sum   = 0.0
                ep_rwd_viol_sum  = 0.0
                ep_active_sum    = 0.0
                # ──────────────────────────────────────────────────────────

                for _ in range(max_steps):
                    action, log_prob, value = agent.act(obs)
                    next_obs, reward, done, truncated, info = env.step(action)
                    terminal = bool(done or truncated)

                    # Buffer stores: obs, action, log_prob, reward, done, value
                    agent.buffer.add(obs, action, log_prob, float(reward), terminal, value)

                    ep_reward  += float(reward)
                    step_count += 1
                    obs         = next_obs
                    decoded     = info.get("decoded_obs", decoded)

                    # Mid-episode update if buffer reaches n_steps
                    if len(agent.buffer) >= ppo_cfg.n_steps:
                        _, _, lv = agent.act(obs)
                        upd = agent.update(last_value=lv)
                        if upd.get("loss", 0.0) > 0.0:
                            ep_losses.append(upd["loss"])

                    # ── Accumulate metrics ─────────────────────────────────
                    if decoded is not None:
                        extra_json = info.get("extra_json")
                        if extra_json is None and len(obs) >= 18:
                            extra_json = {
                                "demand_active": [
                                    int(round(float(obs[15]))),
                                    int(round(float(obs[16]))),
                                    int(round(float(obs[17]))),
                                ]
                            }

                        sla_rates = compute_sla_rates(decoded, cfg, extra_json)
                        ep_sla_sum       += sla_rates["overall"]
                        ep_sla_embb_sum  += sla_rates["embb"]
                        ep_sla_urllc_sum += sla_rates["urllc"]
                        ep_sla_mmtc_sum  += sla_rates["mmtc"]

                        ep_embb_sum      += float(decoded["throughput"].get("eMBB",  0.0)) * float(cfg["env"]["max_thr_mbps"]["eMBB"])
                        ep_urllc_thr_sum += float(decoded["throughput"].get("URLLC", 0.0)) * float(cfg["env"]["max_thr_mbps"]["URLLC"])
                        ep_mmtc_sum      += float(decoded["throughput"].get("mMTC",  0.0)) * float(cfg["env"]["max_thr_mbps"]["mMTC"])

                        lat_ms_raw = (extra_json or {}).get("lat_ms")

                        def _accum_lat(key, idx, ls, ln, _):
                            thr = float(decoded["throughput"].get(key, 0.0)) * float(cfg["env"]["max_thr_mbps"][key])
                            if thr < 0.001:
                                return ls, ln
                            if lat_ms_raw:
                                return ls + float(lat_ms_raw[idx]), ln + 1
                            lat_n = float(decoded["latency"].get(key, 0.0))
                            return ls + lat_n * 2.0 * float(cfg["env"]["max_lat_ms"][key]), ln + 1

                        ep_embb_lat_sum,  ep_embb_lat_n  = _accum_lat("eMBB",  0, ep_embb_lat_sum,  ep_embb_lat_n,  None)
                        ep_urllc_lat_sum, ep_urllc_lat_n = _accum_lat("URLLC", 1, ep_urllc_lat_sum, ep_urllc_lat_n, None)
                        ep_mmtc_lat_sum,  ep_mmtc_lat_n  = _accum_lat("mMTC",  2, ep_mmtc_lat_sum,  ep_mmtc_lat_n,  None)

                        if extra_json and isinstance(extra_json.get("hol_norm"), list):
                            ep_hol_embb_sum  += float(extra_json["hol_norm"][0])
                            ep_hol_urllc_sum += float(extra_json["hol_norm"][1])
                            ep_hol_mmtc_sum  += float(extra_json["hol_norm"][2])
                        elif len(obs) >= 12:
                            ep_hol_embb_sum  += float(obs[9])
                            ep_hol_urllc_sum += float(obs[10])
                            ep_hol_mmtc_sum  += float(obs[11])

                        if len(obs) >= 15:
                            ep_eff_embb_sum  += float(obs[12])
                            ep_eff_urllc_sum += float(obs[13])
                            ep_eff_mmtc_sum  += float(obs[14])

                        rt = (extra_json or {}).get("reward_terms", {})
                        if rt:
                            ep_rwd_thr_sum  += float(rt.get("thr_norm_avg",    0.0))
                            ep_rwd_sla_sum  += float(rt.get("sla_margin_norm", 0.0))
                            ep_rwd_eff_sum  += float(rt.get("eff_norm",        0.0))
                            ep_rwd_viol_sum += float(rt.get("violation_rate",  0.0))
                            ep_active_sum   += float(rt.get("active_slices",   0.0))

                    mean_loss_so_far = float(np.mean(ep_losses)) if ep_losses else 0.0
                    mon.step(
                        step_idx    = step_count,
                        ep_reward   = ep_reward,
                        loss        = mean_loss_so_far if ep_losses else None,
                        obs         = obs,
                        decoded_obs = decoded if isinstance(decoded, dict) else None,
                    )

                    if terminal:
                        sim_done = True
                        break

                step_bar.close()

                # End-of-episode update (remaining buffer)
                if len(agent.buffer) > 0:
                    _, _, lv = agent.act(obs)
                    upd = agent.update(last_value=lv if not sim_done else 0.0)
                    if upd.get("loss", 0.0) > 0.0:
                        ep_losses.append(upd["loss"])

                # ── Episode averages ──────────────────────────────────────
                n = max(1, step_count)

                sla_rate  = ep_sla_sum       / n
                sla_embb  = ep_sla_embb_sum  / n
                sla_urllc = ep_sla_urllc_sum / n
                sla_mmtc  = ep_sla_mmtc_sum  / n
                embb_thr  = ep_embb_sum      / n
                urllc_thr = ep_urllc_thr_sum / n
                mmtc_thr  = ep_mmtc_sum      / n
                embb_lat  = (ep_embb_lat_sum  / ep_embb_lat_n)  if ep_embb_lat_n  > 0 else float("nan")
                urllc_lat = (ep_urllc_lat_sum / ep_urllc_lat_n) if ep_urllc_lat_n > 0 else float("nan")
                mmtc_lat  = (ep_mmtc_lat_sum  / ep_mmtc_lat_n)  if ep_mmtc_lat_n  > 0 else float("nan")
                hol_embb  = ep_hol_embb_sum  / n
                hol_urllc = ep_hol_urllc_sum / n
                hol_mmtc  = ep_hol_mmtc_sum  / n
                eff_embb  = ep_eff_embb_sum  / n
                eff_urllc = ep_eff_urllc_sum / n
                eff_mmtc  = ep_eff_mmtc_sum  / n
                rwd_thr_norm       = ep_rwd_thr_sum  / n
                rwd_sla_margin     = ep_rwd_sla_sum  / n
                rwd_eff_norm       = ep_rwd_eff_sum  / n
                rwd_violation_rate = ep_rwd_viol_sum / n
                active_slices_avg  = ep_active_sum   / n
                mean_loss = float(np.mean(ep_losses)) if ep_losses else 0.0

                if decoded is None:
                    decoded = {"prb_frac": {}}
                prb_frac = decoded.get("prb_frac", {})

                reward_history.append(ep_reward)
                reward_mean10 = sum(reward_history) / len(reward_history)

                mon.end_episode(
                    ep_reward   = ep_reward,
                    sla_rate    = sla_rate,
                    embb_thr    = embb_thr,
                    urllc_lat   = urllc_lat,
                    decoded_obs = decoded,
                    mean_loss   = mean_loss,
                    buf_pct     = None,
                    extra_metrics = {
                        "sla/embb":               sla_embb,
                        "sla/urllc":              sla_urllc,
                        "sla/mmtc":               sla_mmtc,
                        "throughput/URLLC_Mbps":  urllc_thr,
                        "throughput/mMTC_Mbps":   mmtc_thr,
                        "latency/eMBB_ms":        embb_lat  if embb_lat  == embb_lat  else 0.0,
                        "latency/mMTC_ms":        mmtc_lat  if mmtc_lat  == mmtc_lat  else 0.0,
                        "hol/embb":               hol_embb,
                        "hol/urllc":              hol_urllc,
                        "hol/mmtc":               hol_mmtc,
                        "efficiency/embb":        eff_embb,
                        "efficiency/urllc":       eff_urllc,
                        "efficiency/mmtc":        eff_mmtc,
                        "reward_terms/thr_norm":       rwd_thr_norm,
                        "reward_terms/sla_margin":     rwd_sla_margin,
                        "reward_terms/eff_norm":       rwd_eff_norm,
                        "reward_terms/violation_rate": rwd_violation_rate,
                        "reward_terms/active_slices":  active_slices_avg,
                        "reward/mean10":           reward_mean10,
                    },
                )

                # ── JSONL — flat keys, pandas-ready ──────────────────────
                rec = {
                    "episode":      ep,
                    "steps":        step_count,
                    "reward":       round(ep_reward,    6),
                    "reward_mean10":round(reward_mean10,6),
                    "sla_rate":     round(sla_rate,     6),
                    "sla_embb":     round(sla_embb,     6),
                    "sla_urllc":    round(sla_urllc,    6),
                    "sla_mmtc":     round(sla_mmtc,     6),
                    "embb_thr":     round(embb_thr,     6),
                    "urllc_thr":    round(urllc_thr,    6),
                    "mmtc_thr":     round(mmtc_thr,     6),
                    "embb_lat":     nan_or_round(embb_lat),
                    "urllc_lat":    nan_or_round(urllc_lat),
                    "mmtc_lat":     nan_or_round(mmtc_lat),
                    "hol_embb":     round(hol_embb,     6),
                    "hol_urllc":    round(hol_urllc,    6),
                    "hol_mmtc":     round(hol_mmtc,     6),
                    "prb_embb":     max(1, round(prb_frac.get("eMBB",  0.40) * 25)),
                    "prb_urllc":    max(1, round(prb_frac.get("URLLC", 0.32) * 25)),
                    "prb_mmtc":     max(1, round(prb_frac.get("mMTC",  0.28) * 25)),
                    "eff_embb":     round(eff_embb,     6),
                    "eff_urllc":    round(eff_urllc,    6),
                    "eff_mmtc":     round(eff_mmtc,     6),
                    "rwd_thr_norm":       round(rwd_thr_norm,       6),
                    "rwd_sla_margin":     round(rwd_sla_margin,     6),
                    "rwd_eff_norm":       round(rwd_eff_norm,       6),
                    "rwd_violation_rate": round(rwd_violation_rate, 6),
                    "active_slices_avg":  round(active_slices_avg,  6),
                    "mean_loss":    round(mean_loss,    6),
                }
                logf.write(json.dumps(rec) + "\n")
                logf.flush()

                if ep % save_every == 0:
                    agent.save(model_dir / f"ppo_ep{ep}.pt")
                    print(f"[train_ppo.py] Checkpoint saved: ppo_ep{ep}.pt")

                if sim_done:
                    print(f"[train_ppo.py] NS-3 ended after episode {ep}. Increase SIM_TIME.")
                    break

    agent.save(model_dir / "ppo_final.pt")
    env.close()


if __name__ == "__main__":
    main()
