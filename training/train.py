#!/usr/bin/env python3
"""Training entrypoint — DQN.

KEY ARCHITECTURE NOTE (ns3-gym protocol):
  env.reset() called ONCE before the episode loop.
  Episodes are delimited by max_steps steps, not by NS-3 socket resets.
  The NS-3 simulation runs continuously across all episodes of a single launch.

GPU support:
  Automatically uses CUDA if available. GTX 1650 supported.
  NS-3 always runs on CPU — only PyTorch training benefits from GPU.

Resume support:
  --resume: loads latest dqn_ep*.pt checkpoint and continues from that episode.
  Optimizer state, replay buffer, and epsilon are all restored.

SLA rate definition (must match C++ reward in slice-env.cc ScheduleStep):
  A slice is counted as "satisfied" if:
    (a) its observed throughput >= min_thr AND latency observation < 0.5, OR
    (b) it is inactive (mMTC or URLLC with thr < 0.001 Mbps — off-period).
  Off-period exclusion mirrors the C++ inactive-slice logic exactly.

Metric logging (P0-2 fix — episode averages):
  embb_thr, urllc_thr, mmtc_thr, urllc_lat, and sla_rate are episode averages
  computed by accumulating step-level values inside the loop and dividing by
  step_count. This makes logged training metrics directly comparable to the
  episode-average metrics produced by evaluate.py. Previously these were
  last-step snapshots, which were noisy and scientifically incomparable to
  evaluation results.
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
MODEL_DIR = PROJECT_ROOT / "results" / "models"

from agents.dqn.dqn_agent import DqnAgent, DqnConfig
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
    """Compute the fraction of slices meeting their SLA at a single step.

    This function is the Python-side measurement instrument for SLA compliance.
    It MUST remain consistent with the C++ reward function in slice-env.cc.
    Call it once per step and average the results for episode-level logging.

    Inactive-slice rule (mirrors C++ exactly):
      mMTC and URLLC are treated as satisfied when thr < 0.001 Mbps because
      their OnOff traffic models generate genuine silent off-periods that are
      not scheduling failures.

    Latency boundary:
      obs[6:9] is normalised by 2 * maxLatMs in C++, so the SLA boundary
      sits at lat_norm = 0.5, not 1.0.
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


def find_latest_checkpoint(model_dir: Path) -> Path | None:
    """Return the most advanced DQN checkpoint by total_steps count."""
    checkpoints = sorted(
        model_dir.glob("dqn_ep*.pt"),
        key=lambda p: int(p.stem.replace("dqn_ep", ""))
    )
    final = model_dir / "dqn_final.pt"

    if final.exists():
        ckpt = torch.load(final, map_location="cpu")
        final_steps = ckpt.get("total_steps", -1)
        if checkpoints:
            ckpt_steps = torch.load(
                checkpoints[-1], map_location="cpu"
            ).get("total_steps", -1)
            return final if final_steps >= ckpt_steps else checkpoints[-1]
        return final

    return checkpoints[-1] if checkpoints else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent",    type=str,  default="dqn", choices=["dqn"])
    parser.add_argument("--port",     type=int,  default=5555)
    parser.add_argument("--seed",     type=int,  default=42)
    parser.add_argument("--episodes", type=int,  default=None)
    parser.add_argument("--config",   type=str,  default="configs/config.yaml")
    parser.add_argument("--resume",   action="store_true",
                        help="Load latest checkpoint and continue from that episode")
    parser.add_argument("--device",   type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    if args.episodes is None:
        args.episodes = int(cfg["train"]["episodes"])

    set_seed(args.seed)

    device = torch.device(args.device)
    print(f"[train.py] Using device: {device}")
    if device.type == "cuda":
        print(f"[train.py] GPU: {torch.cuda.get_device_name(0)}")

    env = SliceGymEnv(port=args.port, sim_seed=args.seed, start_sim=False)

    # All hyperparameters read explicitly from config.yaml.
    # The DqnConfig dataclass defaults were historically out of sync with
    # config.yaml — this explicit construction is the authoritative source.
    dqn_cfg = DqnConfig(
        obs_dim         = 15,
        action_dim      = 27,
        lr              = float(cfg["dqn"]["lr"]),
        gamma           = float(cfg["dqn"]["gamma"]),
        batch_size      = int(  cfg["dqn"]["batch_size"]),
        buffer_size     = int(  cfg["dqn"]["buffer_size"]),
        tau             = float(cfg["dqn"]["tau"]),
        eps_start       = float(cfg["dqn"]["eps_start"]),
        eps_end         = float(cfg["dqn"]["eps_end"]),
        eps_decay_steps = int(  cfg["dqn"]["eps_decay_steps"]),
        grad_clip       = float(cfg["dqn"]["grad_clip"]),
    )
    agent    = DqnAgent(dqn_cfg, device)
    start_ep = 1

    # ── Resume ────────────────────────────────────────────────────────────────
    model_dir = MODEL_DIR
    os.makedirs(model_dir, exist_ok=True)
    print(f"[train.py] Model directory: {model_dir.resolve()}")

    if args.resume:
        ckpt_path = find_latest_checkpoint(model_dir)
        if ckpt_path is not None:
            print(f"[train.py] Resuming from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device)
            agent.online_net.load_state_dict(ckpt["online"])
            agent.target_net.load_state_dict(ckpt["target"])
            agent.optim.load_state_dict(ckpt["optim"])
            agent.total_steps = ckpt.get("total_steps", 0)
            agent.load_buffer(ckpt_path)
            if "dqn_ep" in ckpt_path.stem:
                ckpt_ep  = int(ckpt_path.stem.replace("dqn_ep", "")) + 1
                log_path = PROJECT_ROOT / "results" / "logs" / "dqn_log.jsonl"
                if log_path.exists():
                    with log_path.open() as f:
                        last = None
                        for line in f:
                            line = line.strip()
                            if line:
                                last = line
                    log_ep   = (json.loads(last)["episode"] + 1) if last else ckpt_ep
                    start_ep = max(ckpt_ep, log_ep)
                else:
                    start_ep = ckpt_ep
            else:
                log_path = PROJECT_ROOT / "results" / "logs" / "dqn_log.jsonl"
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
            print(f"[train.py] Continuing from episode {start_ep} "
                  f"(total_steps={agent.total_steps}, eps={agent.epsilon():.3f})")
        else:
            print(f"[train.py] No checkpoint found in {model_dir.resolve()}")
            print("[train.py] Starting from scratch.")
    # ── End Resume ─────────────────────────────────────────────────────────────

    log_path = PROJECT_ROOT / "results" / "logs" / "dqn_log.jsonl"
    os.makedirs(log_path.parent, exist_ok=True)

    max_steps  = int(cfg["train"]["max_steps"])
    save_every = int(cfg["train"]["save_every"])

    print(f"[train.py] Connecting to NS-3 on port {args.port}...")
    obs, info = env.reset()
    print("[train.py] Connected. Starting training.")

    # P0-1 verification: after the maxUes fix, obs[12:15] should read
    # approximately 0.500, 0.500, 0.400 — not 1.0, 1.0, 1.0.
    print(f"[train.py] obs[12:15] = {obs[12]:.3f}, {obs[13]:.3f}, {obs[14]:.3f}  "
          f"(expected ~0.500, 0.500, 0.400 — confirms P0-1 fix is live)")

    decoded = info.get("decoded_obs", None)

    with TrainingMonitor("dqn", start_ep + args.episodes - 1, max_steps,
                         PROJECT_ROOT / "results") as mon:
        with log_path.open("a", encoding="utf-8") as logf:
            for ep in range(start_ep, start_ep + args.episodes):
                ep_reward  = 0.0
                step_count = 0
                sim_done   = False
                step_bar   = mon.begin_episode(ep)

                ep_losses: list[float] = []

                # ── P0-2 accumulators — one value per step, averaged at end ──
                # The key insight: ep_reward was already accumulated correctly
                # by summing reward each step. We extend the same pattern to
                # the three logged performance metrics so they become episode
                # averages rather than last-step snapshots.
                ep_sla_sum       = 0.0
                ep_embb_sum      = 0.0
                ep_urllc_thr_sum = 0.0
                ep_mmtc_sum      = 0.0
                ep_urllc_lat_sum = 0.0
                # ─────────────────────────────────────────────────────────────

                for _ in range(max_steps):
                    action = agent.act(obs, explore=True)
                    next_obs, reward, done, truncated, info = env.step(action)
                    terminal = bool(done or truncated)

                    agent.store(obs, action, reward, next_obs, terminal)
                    train_info = agent.train_step()
                    if isinstance(train_info, dict) and train_info.get("loss", 0.0) > 0.0:
                        ep_losses.append(train_info["loss"])

                    ep_reward  += float(reward)
                    step_count += 1
                    obs         = next_obs
                    decoded     = info.get("decoded_obs", decoded)

                    # Accumulate per-step metrics in physical units (Mbps, ms).
                    # Denormalisation happens here so accumulators hold real values.
                    # Latency: obs_val = actual_ms / (2 * maxLatMs)
                    #   → actual_ms = obs_val * 2 * maxLatMs
                    if decoded is not None:
                        ep_sla_sum       += compute_sla_rate(decoded, cfg)
                        ep_embb_sum      += (float(decoded["throughput"].get("eMBB",  0.0))
                                             * float(cfg["env"]["max_thr_mbps"]["eMBB"]))
                        ep_urllc_thr_sum += (float(decoded["throughput"].get("URLLC", 0.0))
                                             * float(cfg["env"]["max_thr_mbps"]["URLLC"]))
                        ep_mmtc_sum      += (float(decoded["throughput"].get("mMTC",  0.0))
                                             * float(cfg["env"]["max_thr_mbps"]["mMTC"]))
                        ep_urllc_lat_sum += (float(decoded["latency"].get("URLLC", 0.0))
                                             * 2.0
                                             * float(cfg["env"]["max_lat_ms"]["URLLC"]))

                    mean_loss_so_far = float(np.mean(ep_losses)) if ep_losses else 0.0
                    mon.step(
                        ep_reward = ep_reward,
                        epsilon   = agent.epsilon(),
                        loss      = mean_loss_so_far if ep_losses else None,
                        buf_pct   = len(agent.buffer) / agent.cfg.buffer_size,
                        sla_rate  = None,
                    )

                    if terminal:
                        sim_done = True
                        break

                step_bar.close()

                # ── P0-2: compute episode averages from accumulators ──────────
                n         = max(1, step_count)
                sla_rate  = ep_sla_sum       / n
                embb_thr  = ep_embb_sum      / n
                urllc_thr = ep_urllc_thr_sum / n
                mmtc_thr  = ep_mmtc_sum      / n
                urllc_lat = ep_urllc_lat_sum  / n
                # ─────────────────────────────────────────────────────────────

                # PRB allocation: last-step snapshot is correct here.
                # PRBs are the agent's control output — the final allocation
                # after 1000 steps of adjustment, not a metric to be averaged.
                if decoded is None:
                    decoded = {"prb_frac": {}}
                prb_frac  = decoded.get("prb_frac", {})
                mean_loss = float(np.mean(ep_losses)) if ep_losses else 0.0

                mon.end_episode(
                    ep_reward   = ep_reward,
                    sla_rate    = sla_rate,
                    embb_thr    = embb_thr,
                    urllc_lat   = urllc_lat,
                    decoded_obs = decoded,
                    epsilon     = agent.epsilon(),
                    mean_loss   = mean_loss,
                )

                # JSONL log. All throughput, latency, and sla_rate values are
                # episode averages as of the P0-2 fix. PRB values are last-step.
                rec = {
                    "episode":     ep,
                    "reward":      round(ep_reward,  6),
                    "steps":       step_count,
                    "embb_thr":    round(embb_thr,   6),   # avg Mbps
                    "urllc_thr":   round(urllc_thr,  6),   # avg Mbps
                    "mmtc_thr":    round(mmtc_thr,   6),   # avg Mbps
                    "urllc_lat":   round(urllc_lat,  6),   # avg ms
                    "sla_rate":    round(sla_rate,   6),   # avg fraction
                    "prb_embb":    max(1, round(prb_frac.get("eMBB",  0.4)  * 25)),
                    "prb_urllc":   max(1, round(prb_frac.get("URLLC", 0.32) * 25)),
                    "prb_mmtc":    max(1, round(prb_frac.get("mMTC",  0.28) * 25)),
                    "epsilon":     round(agent.epsilon(), 6),
                    "mean_loss":   round(mean_loss,   6),
                    "buffer_size": len(agent.buffer),
                }
                logf.write(json.dumps(rec) + "\n")
                logf.flush()

                if ep % save_every == 0:
                    agent.save(model_dir / f"dqn_ep{ep}.pt")
                    print(f"[train.py] Checkpoint saved: dqn_ep{ep}.pt")

                if sim_done:
                    print(f"[train.py] NS-3 ended after episode {ep}. Increase SIM_TIME.")
                    break

    agent.save(model_dir / "dqn_final.pt")
    env.close()


if __name__ == "__main__":
    main()
