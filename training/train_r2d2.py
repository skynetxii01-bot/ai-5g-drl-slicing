#!/usr/bin/env python3
"""Training entrypoint — R2D2.

Fixes vs previous version:
  - Shape error: overlap buffer now stores only raw (unpadded) numpy arrays.
    collect_sequence always pads from a clean list — no mixed shapes possible.
  - GPU: automatically uses CUDA if available (GTX 1650 supported).
  - Resume: --resume flag loads latest checkpoint and continues from that episode.

ns3-gym architecture note:
  env.reset() called ONCE before the loop.
  Episodes are delimited by max_steps steps, not by NS-3 socket resets.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
model_dir = PROJECT_ROOT / "results" / "models"

from agents.r2d2.r2d2_agent import R2D2Agent, R2D2Config
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


def collect_sequence(
    obs_buf:  List[np.ndarray],
    act_buf:  List[int],
    rew_buf:  List[float],
    done_buf: List[bool],
    next_buf: List[np.ndarray],
    seq_len:  int,
) -> Dict[str, np.ndarray]:
    """
    Pack a variable-length list of transitions into fixed-length arrays.

    All inputs are plain Python lists of raw numpy arrays (shape=(15,)).
    Padding is added at the END to reach seq_len.
    No mixed shapes are possible because we never store padded arrays
    back into the overlap buffer.
    """
    n   = min(len(obs_buf), seq_len)
    pad = max(0, seq_len - n)

    zero_obs = np.zeros_like(obs_buf[0])

    obs_arr  = np.stack(obs_buf  + [zero_obs] * pad)
    nxt_arr  = np.stack(next_buf + [zero_obs] * pad)
    act_arr  = np.array(act_buf  + [0]        * pad, dtype=np.int64)
    rew_arr  = np.array(rew_buf  + [0.0]      * pad, dtype=np.float32)
    done_arr = np.array(done_buf + [True]     * pad, dtype=np.float32)

    return dict(obs=obs_arr, next_obs=nxt_arr, action=act_arr,
                reward=rew_arr, done=done_arr)


def find_latest_checkpoint(model_dir: Path) -> Path | None:
    checkpoints = sorted(
        model_dir.glob("r2d2_ep*.pt"),
        key=lambda p: int(p.stem.replace("r2d2_ep", ""))
    )
    final = model_dir / "r2d2_final.pt"

    if final.exists():
        ckpt = torch.load(final, map_location="cpu")
        final_steps = ckpt.get("train_steps", -1)
        ckpt_steps  = torch.load(checkpoints[-1], map_location="cpu").get("train_steps", -1) if checkpoints else -1
        return final if final_steps >= ckpt_steps else checkpoints[-1]

    return checkpoints[-1] if checkpoints else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",     type=int,  default=5555)
    parser.add_argument("--seed",     type=int,  default=42)
    parser.add_argument("--episodes", type=int,  default=None)
    parser.add_argument("--config",   type=str,  default="configs/config.yaml")
    parser.add_argument("--resume",   action="store_true",
                        help="Load latest checkpoint and continue training from that episode")
    parser.add_argument("--device",   type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg      = load_config(Path(args.config))
    r2d2_raw = cfg.get("r2d2", {})
    if args.episodes is None:
        args.episodes = int(cfg["train"]["episodes"])

    set_seed(args.seed)

    device = torch.device(args.device)
    print(f"[train_r2d2.py] Using device: {device}")
    if device.type == "cuda":
        print(f"[train_r2d2.py] GPU: {torch.cuda.get_device_name(0)}")

    env = SliceGymEnv(port=args.port, sim_seed=args.seed, start_sim=False)

    r2d2_cfg = R2D2Config(
        obs_dim=15, action_dim=27,
        lr             = float(r2d2_raw.get("lr",                1e-4)),
        gamma          = float(r2d2_raw.get("gamma",             0.99)),
        batch_size     = int(  r2d2_raw.get("batch_size",        32)),
        seq_len        = int(  r2d2_raw.get("seq_len",           40)),
        burn_in        = int(  r2d2_raw.get("burn_in",           40)),
        n_step         = int(  r2d2_raw.get("n_step",            5)),
        per_alpha      = float(r2d2_raw.get("per_alpha",         0.6)),
        per_beta_start = float(r2d2_raw.get("per_beta_start",    0.4)),
        per_beta_end   = float(r2d2_raw.get("per_beta_end",      1.0)),
        buffer_size    = int(  r2d2_raw.get("buffer_size",       100_000)),
        grad_clip      = float(r2d2_raw.get("grad_clip",         40.0)),
    )
    agent = R2D2Agent(r2d2_cfg, device)
    target_update_every = int(r2d2_raw.get("target_update_every", 100))

    log_path  = PROJECT_ROOT / "results" / "logs" / "r2d2_log.jsonl"
    model_dir = PROJECT_ROOT / "results" / "models"
    os.makedirs(log_path.parent, exist_ok=True)
    os.makedirs(model_dir,       exist_ok=True)

    max_steps  = int(cfg["train"]["max_steps"])
    save_every = int(cfg["train"]["save_every"])
    seq_len    = r2d2_cfg.seq_len
    start_ep   = 1

    # ── Resume ────────────────────────────────────────────────────────────────
    os.makedirs(model_dir, exist_ok=True)
    print(f"[train_r2d2.py] Model directory: {model_dir.resolve()}")

    if args.resume:
        ckpt_path = find_latest_checkpoint(model_dir)
        if ckpt_path is not None:
            print(f"[train_r2d2.py] Resuming from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device)
            agent.online.load_state_dict(ckpt["online"])
            agent.target.load_state_dict(ckpt["target"])
            agent.train_steps = ckpt.get("train_steps", 0)
            agent.optim.load_state_dict(ckpt["optim"])
            if "r2d2_ep" in ckpt_path.stem:
                ckpt_ep = int(ckpt_path.stem.replace("r2d2_ep", "")) + 1
                log_path = PROJECT_ROOT / "results" / "logs" / "r2d2_log.jsonl"
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
                log_path = PROJECT_ROOT / "results" / "logs" / "r2d2_log.jsonl"
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
            print(f"[train_r2d2.py] Continuing from episode {start_ep}")
        else:
            print(f"[train_r2d2.py] No checkpoint found in {model_dir.resolve()}")
    # ── End Resume ─────────────────────────────────────────────────────────────

    print(f"[train_r2d2.py] Connecting to NS-3 on port {args.port}...")
    obs, info = env.reset()
    print("[train_r2d2.py] Connected. Starting training.")
    decoded = info.get("decoded_obs", None)

    with TrainingMonitor("r2d2", start_ep + args.episodes - 1, max_steps,
                         PROJECT_ROOT / "results") as mon:
        with log_path.open("a", encoding="utf-8") as logf:
            for ep in range(start_ep, start_ep + args.episodes):
                ep_reward  = 0.0
                step_count = 0
                sim_done   = False
                hidden     = None

                obs_buf:  List[np.ndarray] = []
                act_buf:  List[int]        = []
                rew_buf:  List[float]      = []
                done_buf: List[bool]       = []
                next_buf: List[np.ndarray] = []

                step_bar = mon.begin_episode(ep)
                ep_losses: list[float] = []

                for _ in range(max_steps):
                    action, hidden = agent.act(obs, hidden=hidden, explore=True)
                    next_obs, reward, done, truncated, info = env.step(action)
                    terminal = bool(done or truncated)

                    obs_buf.append(obs.copy())
                    act_buf.append(action)
                    rew_buf.append(float(reward))
                    done_buf.append(terminal)
                    next_buf.append(next_obs.copy())

                    ep_reward  += float(reward)
                    step_count += 1
                    obs         = next_obs
                    decoded     = info.get("decoded_obs", decoded)
                    mean_loss_so_far = float(np.mean(ep_losses)) if ep_losses else 0.0
                    mon.step(
                        ep_reward = ep_reward,
                        loss      = mean_loss_so_far if ep_losses else None,
                        buf_pct   = len(agent.buffer) / r2d2_cfg.buffer_size,
                    )

                    if len(obs_buf) >= seq_len or terminal:
                        seq = collect_sequence(
                            obs_buf, act_buf, rew_buf, done_buf, next_buf, seq_len
                        )
                        agent.add_sequence(
                            obs      = seq["obs"],
                            action   = seq["action"],
                            reward   = seq["reward"],
                            done     = seq["done"],
                            next_obs = seq["next_obs"],
                            priority = 1.0,
                        )
                        overlap  = min(r2d2_cfg.burn_in, seq_len - 1)
                        obs_buf  = obs_buf[-overlap:]
                        act_buf  = act_buf[-overlap:]
                        rew_buf  = rew_buf[-overlap:]
                        done_buf = done_buf[-overlap:]
                        next_buf = next_buf[-overlap:]

                    train_info = agent.train_step()
                    if isinstance(train_info, dict) and train_info.get("loss", 0.0) > 0.0:
                        ep_losses.append(train_info["loss"])

                    if agent.train_steps % target_update_every == 0:
                        agent.sync_target()

                    if terminal:
                        sim_done = True
                        break

                step_bar.close()

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
                    train_steps = agent.train_steps,
                    mean_loss   = mean_loss,
                )
                # JSONL log
                rec = {
                    "episode":      ep,
                    "reward":       round(ep_reward,         6),
                    "steps":        step_count,
                    "embb_thr":     round(embb_thr,          6),
                    "urllc_thr":    round(urllc_thr,         6),
                    "mmtc_thr":     round(mmtc_thr,          6),
                    "urllc_lat":    round(urllc_lat,         6),
                    "sla_rate":     round(sla_rate,          6),
                    "prb_embb":     max(1, round(prb_frac.get("eMBB",  0.4)  * 25)),
                    "prb_urllc":    max(1, round(prb_frac.get("URLLC", 0.32) * 25)),
                    "prb_mmtc":     max(1, round(prb_frac.get("mMTC",  0.28) * 25)),
                    "train_steps":  agent.train_steps,
                    "mean_loss":    round(mean_loss,         6),
                }
                logf.write(json.dumps(rec) + "\n")
                logf.flush()

                if ep % save_every == 0:
                    agent.save(model_dir / f"r2d2_ep{ep}.pt")
                    print(f"[train_r2d2.py] Checkpoint saved: r2d2_ep{ep}.pt")

                if sim_done:
                    print(f"[train_r2d2.py] NS-3 ended after episode {ep}. Increase SIM_TIME.")
                    break

    agent.save(model_dir / "r2d2_final.pt")
    env.close()


if __name__ == "__main__":
    main()