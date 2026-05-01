#!/usr/bin/env python3
"""Training entrypoint — PPO.

GPU support and --resume flag added.
ns3-gym single-reset architecture — see train.py for details.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
model_dir = PROJECT_ROOT / "results" / "models"

from agents.ppo.ppo_agent import PpoAgent, PpoConfig
from envs.slice_gym_env import SLICE_NAMES, SliceGymEnv
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


def compute_sla_rate(decoded_obs: Dict, cfg: Dict) -> float:
    max_thr = cfg["env"]["max_thr_mbps"]
    min_thr = cfg["env"]["min_thr_mbps"]
    sat = 0
    for s in SLICE_NAMES:
        thr = float(decoded_obs["throughput"][s]) * float(max_thr[s])
        lat_norm = float(decoded_obs["latency"][s])
        if thr >= float(min_thr[s]) and lat_norm < 1.0:
            sat += 1
    return sat / len(SLICE_NAMES)


def find_latest_checkpoint(model_dir: Path) -> Path | None:
    checkpoints = sorted(
        model_dir.glob("ppo_ep*.pt"),
        key=lambda p: int(p.stem.replace("ppo_ep", ""))
    )
    final = model_dir / "ppo_final.pt"

    if final.exists():
        # PPO has no step counter — compare by episode number in filename
        ckpt_ep = int(checkpoints[-1].stem.replace("ppo_ep", "")) if checkpoints else -1
        # final.pt is always saved last, so prefer it if any numbered ckpt exists
        return final
    
    return checkpoints[-1] if checkpoints else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",     type=int,  default=5555)
    parser.add_argument("--seed",     type=int,  default=42)
    parser.add_argument("--episodes", type=int,  default=None)
    parser.add_argument("--config",   type=str,  default="configs/config.yaml")
    parser.add_argument("--resume",   action="store_true",
                        help="Load latest checkpoint and continue from that episode")
    parser.add_argument("--device",   type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg     = load_config(Path(args.config))
    ppo_raw = cfg.get("ppo", {})
    if args.episodes is None:
        args.episodes = int(cfg["train"]["episodes"])

    set_seed(args.seed)

    device = torch.device(args.device)
    print(f"[train_ppo.py] Using device: {device}")
    if device.type == "cuda":
        print(f"[train_ppo.py] GPU: {torch.cuda.get_device_name(0)}")

    env = SliceGymEnv(port=args.port, sim_seed=args.seed, start_sim=False)

    ppo_cfg = PpoConfig(
        obs_dim=15, action_dim=27,
        lr         = float(ppo_raw.get("lr",         3e-4)),
        gamma      = float(ppo_raw.get("gamma",      0.99)),
        lambda_gae = float(ppo_raw.get("lambda_gae", 0.95)),
        eps_clip   = float(ppo_raw.get("eps_clip",   0.2)),
        n_steps    = int(  ppo_raw.get("n_steps",    2048)),
        epochs     = int(  ppo_raw.get("epochs",     4)),
        batch_size = int(  ppo_raw.get("batch_size", 64)),
        ent_coef   = float(ppo_raw.get("ent_coef",   0.01)),
        vf_coef    = float(ppo_raw.get("vf_coef",    0.5)),
    )
    agent    = PpoAgent(ppo_cfg, device)
    start_ep = 1

    # ── Resume ────────────────────────────────────────────────────────────────
    os.makedirs(model_dir, exist_ok=True)
    print(f"[train_ppo.py] Model directory: {model_dir.resolve()}")

    if args.resume:
        ckpt_path = find_latest_checkpoint(model_dir)
        if ckpt_path is not None:
            print(f"[train_ppo.py] Resuming from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device)
            agent.net.load_state_dict(ckpt["model"])
            agent.optim.load_state_dict(ckpt["optim"])
            if "ppo_ep" in ckpt_path.stem:
                ckpt_ep = int(ckpt_path.stem.replace("ppo_ep", "")) + 1
                log_path = PROJECT_ROOT / "results" / "logs" / "ppo_log.jsonl"
                if log_path.exists():
                    with log_path.open() as f:
                        last = None
                        for line in f:
                            line = line.strip()
                            if line:
                                last = line
                    log_ep = (json.loads(last)["episode"] + 1) if last else ckpt_ep
                    start_ep = max(ckpt_ep, log_ep)
                else:
                    start_ep = ckpt_ep
            else:
                log_path = PROJECT_ROOT / "results" / "logs" / "ppo_log.jsonl"
                if log_path.exists():
                    with log_path.open() as f:
                        last = None
                        for line in f:
                            line = line.strip()
                            if line:
                                last = line
                    start_ep = (json.loads(last)["episode"] + 1) if last else 1
                else:
                    start_ep = 1
            print(f"[train_ppo.py] Continuing from episode {start_ep}")
        else:
            print(f"[train_ppo.py] No checkpoint found in {model_dir.resolve()}")
    # ── End Resume ─────────────────────────────────────────────────────────────

    log_path = PROJECT_ROOT / "results" / "logs" / "ppo_log.jsonl"
    os.makedirs(log_path.parent, exist_ok=True)

    max_steps  = int(cfg["train"]["max_steps"])
    save_every = int(cfg["train"]["save_every"])

    print(f"[train_ppo.py] Connecting to NS-3 on port {args.port}...")
    obs, info = env.reset()
    print("[train_ppo.py] Connected. Starting training.")
    decoded = info.get("decoded_obs", None)

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
                for _ in range(max_steps):
                    action, log_prob, value = agent.act(obs)
                    next_obs, reward, done, truncated, info = env.step(action)
                    terminal = bool(done or truncated)

                    agent.buffer.add(obs, action, reward, terminal, log_prob, value)
                    ep_reward  += float(reward)
                    step_count += 1
                    obs         = next_obs
                    decoded     = info.get("decoded_obs", decoded)
                    mean_loss_so_far = float(np.mean(ep_losses)) if ep_losses else 0.0
                    mon.step(
                        ep_reward = ep_reward,
                        loss      = mean_loss_so_far if ep_losses else None,
                    )

                    if len(agent.buffer.steps) >= ppo_cfg.n_steps:
                        _, _, lv = agent.act(obs)
                        upd = agent.update(last_value=lv)
                        if isinstance(upd, dict) and upd.get("loss", 0.0) > 0.0:
                            ep_losses.append(upd["loss"])

                    if terminal:
                        sim_done = True
                        break

                step_bar.close()

                if len(agent.buffer.steps) > 0:
                    upd = agent.update(last_value=0.0)
                    if isinstance(upd, dict) and upd.get("loss", 0.0) > 0.0:
                        ep_losses.append(upd["loss"])

                if decoded is None:
                    decoded = {"throughput": {s: 0.0 for s in SLICE_NAMES},
                               "latency":    {s: 0.0 for s in SLICE_NAMES}}

                embb_thr  = float(decoded["throughput"].get("eMBB",  0.0)) * float(cfg["env"]["max_thr_mbps"]["eMBB"])
                urllc_lat = float(decoded["latency"].get("URLLC",    0.0)) * float(cfg["env"]["max_lat_ms"]["URLLC"])
                sla_rate  = compute_sla_rate(decoded, cfg)

                mean_loss = float(np.mean(ep_losses)) if ep_losses else 0.0
                prb_frac  = decoded.get("prb_frac", {})
                mmtc_thr  = float(decoded["throughput"].get("mMTC",  0.0)) * float(cfg["env"]["max_thr_mbps"]["mMTC"])
                urllc_thr = float(decoded["throughput"].get("URLLC", 0.0)) * float(cfg["env"]["max_thr_mbps"]["URLLC"])

                mon.end_episode(
                    ep_reward   = ep_reward,
                    sla_rate    = sla_rate,
                    embb_thr    = embb_thr,
                    urllc_lat   = urllc_lat,
                    decoded_obs = decoded,
                    mean_loss   = mean_loss,
                )

                # JSONL log
                rec = {
                    "episode":   ep,
                    "reward":    round(ep_reward,  6),
                    "steps":     step_count,
                    "embb_thr":  round(embb_thr,   6),
                    "urllc_thr": round(urllc_thr,  6),
                    "mmtc_thr":  round(mmtc_thr,   6),
                    "urllc_lat": round(urllc_lat,   6),
                    "sla_rate":  round(sla_rate,    6),
                    "prb_embb":  max(1, round(prb_frac.get("eMBB",  0.4)  * 25)),
                    "prb_urllc": max(1, round(prb_frac.get("URLLC", 0.32) * 25)),
                    "prb_mmtc":  max(1, round(prb_frac.get("mMTC",  0.28) * 25)),
                    "mean_loss": round(mean_loss,   6),
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