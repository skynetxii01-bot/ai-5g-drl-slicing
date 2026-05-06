"""
training/monitor.py
Unified training monitor for DQN / PPO / R2D2.

Provides:
  - tqdm step-level progress bar  (x / max_steps per episode)
  - Rich per-episode summary table with PRB allocation bar
  - TensorBoard scalar logging
  - Wall-clock timing and ETA
  - Running reward mean (last 10 episodes)
  - obs[12:15] sla_headroom display (live + episode summary)
  - mean_loss and buffer fill in episode summary table

Usage (DQN):
    from monitor import TrainingMonitor

    with TrainingMonitor("dqn", total_episodes=500,
                         max_steps=1000,
                         log_dir=PROJECT_ROOT / "results") as mon:
        obs, info = env.reset()
        for ep in range(start_ep, start_ep + args.episodes):
            step_bar = mon.begin_episode(ep)
            for _ in range(max_steps):
                action = agent.act(obs, explore=True)
                next_obs, reward, done, truncated, info = env.step(action)
                mon.step(
                    step_idx    = step_count,
                    ep_reward   = ep_reward,
                    epsilon     = agent.epsilon(),
                    loss        = loss,
                    buf_pct     = len(agent.buffer) / agent.cfg.buffer_size,
                    obs         = next_obs,
                    decoded_obs = info.get("decoded_obs"),
                )
                obs = next_obs
                if done or truncated:
                    break
            step_bar.close()
            mon.end_episode(
                ep_reward   = ep_reward,
                sla_rate    = sla_rate,
                embb_thr    = embb_thr,
                urllc_lat   = urllc_lat,
                decoded_obs = decoded,
                epsilon     = agent.epsilon(),   # DQN only
                mean_loss   = mean_loss,
                buf_pct     = len(agent.buffer) / agent.cfg.buffer_size,
            )
"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Dict, Optional

from rich.console import Console
from rich.rule import Rule
from rich.table import Table
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

console = Console()

_PRB_TOTAL = 25   # must match slice-rl-sim.cc totalPrbs

# tqdm colour per agent — makes it easy to distinguish agents at a glance.
_AGENT_COLOUR = {
    "DQN":  "red",
    "PPO":  "green",
    "R2D2": "magenta",
}


class TrainingMonitor:
    """
    Context-manager wrapping one full training run.

    Parameters
    ----------
    agent_name     : "dqn" | "ppo" | "r2d2"  (case-insensitive)
    total_episodes : total episodes in this run  (for ETA)
    max_steps      : max steps per episode       (for tqdm total)
    log_dir        : PROJECT_ROOT / "results"
                     TensorBoard → log_dir/tensorboard/<agent_name>/
    """

    def __init__(
        self,
        agent_name: str,
        total_episodes: int,
        max_steps: int,
        log_dir: Path,
    ) -> None:
        self.agent_name     = agent_name.upper()
        self.total_episodes = total_episodes
        self.max_steps      = max_steps

        tb_path = Path(log_dir) / "tensorboard" / agent_name.lower()
        tb_path.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(tb_path))

        self._ep          : int   = 0
        self._run_start   : float = 0.0
        self._ep_start    : float = 0.0
        self._step_bar    : Optional[tqdm] = None

        # Last-known metrics
        self._ep_reward   : float            = 0.0
        self._sla_rate    : float            = 0.0
        self._embb_thr    : float            = 0.0
        self._urllc_lat   : float            = 0.0
        self._mean_loss   : Optional[float]  = None
        self._buf_pct     : Optional[float]  = None
        self._epsilon     : Optional[float]  = None
        self._train_steps : Optional[int]    = None
        self._prb         : Dict[str, int]   = {"eMBB": 10, "URLLC": 8, "mMTC": 7}

        # Track all 15 obs dims.
        # [0:3]  prb_frac, [3:6] throughput, [6:9] latency,
        # [9:12] hol_delay, [12:15] sla_headroom
        self._last_obs    : list[float]      = [0.0] * 18
        self._last_step   : int              = 0

        # Running reward mean over last 10 episodes (smoothed signal).
        self._reward_history: deque[float] = deque(maxlen=10)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "TrainingMonitor":
        self._run_start = time.time()
        console.print(
            f"\n[bold]{self.agent_name}[/bold] training started — "
            f"{self.total_episodes} episodes × {self.max_steps} steps"
        )
        return self

    def __exit__(self, *_) -> None:
        self.writer.close()
        if self._step_bar is not None:
            self._step_bar.close()

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def begin_episode(self, ep: int) -> tqdm:
        """Call at the START of each episode. Returns the tqdm bar."""
        self._ep       = ep
        self._ep_start = time.time()
        colour = _AGENT_COLOUR.get(self.agent_name, "white")
        self._step_bar = tqdm(
            total         = self.max_steps,
            desc          = f"[{self.agent_name}] ep {ep:>4}/{self.total_episodes}",
            unit          = "step",
            leave         = False,
            dynamic_ncols = True,
            colour        = colour,
        )
        return self._step_bar

    def step(
        self,
        step_idx    : int | None   = None,
        ep_reward   : float        = 0.0,
        epsilon     : float | None = None,
        loss        : float | None = None,
        buf_pct     : float | None = None,
        sla_rate    : float | None = None,
        obs         : Optional[object] = None,
        decoded_obs : Optional[Dict]   = None,
    ) -> None:
        """Advance the tqdm bar by one step. Call once per env.step()."""
        if step_idx is not None:
            self._last_step = step_idx

        # Store full obs vector (18 dims).
        if obs is not None:
            try:
                self._last_obs = [float(obs[i]) for i in range(min(18 , len(obs)))]
            except Exception:
                pass

        if decoded_obs:
            prb_frac = decoded_obs.get("prb_frac", {})
            self._prb = {
                "eMBB":  max(1, round(prb_frac.get("eMBB",  0.40) * _PRB_TOTAL)),
                "URLLC": max(1, round(prb_frac.get("URLLC", 0.32) * _PRB_TOTAL)),
                "mMTC":  max(1, round(prb_frac.get("mMTC",  0.28) * _PRB_TOTAL)),
            }

        if self._step_bar is not None:
            pf: dict = {"r": f"{ep_reward:.2f}"}
            if epsilon  is not None: pf["ε"]    = f"{epsilon:.3f}"
            if loss     is not None: pf["loss"] = f"{loss:.4f}"
            if buf_pct  is not None: pf["buf"]  = f"{buf_pct:.0%}"
            if sla_rate is not None: pf["sla"]  = f"{sla_rate:.0%}"
            pf["prb"] = (
                f"{self._prb['eMBB']}/"
                f"{self._prb['URLLC']}/"
                f"{self._prb['mMTC']}"
            )
            # Show sla_headroom (obs[12:15]) — the fixed feature.
            if len(self._last_obs) >= 15:
                pf["head"] = (
                    f"{self._last_obs[12]:.2f}/"
                    f"{self._last_obs[13]:.2f}/"
                    f"{self._last_obs[14]:.2f}"
                )
            self._step_bar.set_postfix(pf)
            self._step_bar.update(1)

        # Detailed live print every 10% of steps — includes sla_headroom.
        if (
            self._last_step > 0
            and self._last_step % max(1, self.max_steps // 10) == 0
        ):
            head_str = (
                f"{self._last_obs[12]:.3f}/"
                f"{self._last_obs[13]:.3f}/"
                f"{self._last_obs[14]:.3f}"
                if len(self._last_obs) >= 15 else "n/a"
            )
            thr_str = (
                f"{self._last_obs[3]:.3f}/"
                f"{self._last_obs[4]:.3f}/"
                f"{self._last_obs[5]:.3f}"
                if len(self._last_obs) >= 6 else "n/a"
            )
            lat_str = (
                 f"{self._last_obs[6]:.3f}/"
                 f"{self._last_obs[7]:.3f}/"
                 f"{self._last_obs[8]:.3f}"
                 if len(self._last_obs) >= 9 else "n/a"
            )
                
            flag_str = (
                f"{self._last_obs[15]:.0f}/"
                f"{self._last_obs[16]:.0f}/"
                f"{self._last_obs[17]:.0f}"
                if len(self._last_obs) >= 18 else "n/a"
            )
                     
            console.print(
                f"[bold green][LIVE {self.agent_name} "
                f"@ {self._last_step}/{self.max_steps}][/bold green]  "
                f"PRB={self._prb['eMBB']}/{self._prb['URLLC']}/{self._prb['mMTC']}  "
                f"thr_norm={thr_str}  "
                f"sla_head={head_str}  "
                f"lat_norm={lat_str}  "
                f"active={flag_str} "
                f"r={ep_reward:.3f}"
                + (f"  ε={epsilon:.4f}" if epsilon is not None else "")
                + (f"  loss={loss:.4f}" if loss is not None else "")
            )

    def end_episode(
        self,
        ep_reward   : float,
        sla_rate    : float,
        embb_thr    : float,
        urllc_lat   : float,
        decoded_obs : Dict,
        epsilon     : Optional[float] = None,
        train_steps : Optional[int]   = None,
        mean_loss   : Optional[float] = None,
        buf_pct     : Optional[float] = None,
    ) -> None:
        """
        Call AFTER the step loop, before logging to JSONL.

        decoded_obs : info["decoded_obs"] from the last env.step()
        epsilon     : agent.epsilon()      — DQN only, omit for PPO/R2D2
        train_steps : agent.train_steps    — R2D2 only, omit for DQN/PPO
        mean_loss   : mean loss over episode steps
        buf_pct     : replay buffer fill fraction (0–1)
        """
        self._ep_reward   = ep_reward
        self._sla_rate    = sla_rate
        self._embb_thr    = embb_thr
        self._urllc_lat   = urllc_lat
        self._mean_loss   = mean_loss
        self._buf_pct     = buf_pct
        self._epsilon     = epsilon
        self._train_steps = train_steps

        self._reward_history.append(ep_reward)

        # Decode PRB from normalised observation.
        prb_frac = (decoded_obs or {}).get("prb_frac", {})
        self._prb = {
            "eMBB":  max(1, round(prb_frac.get("eMBB",  0.40) * _PRB_TOTAL)),
            "URLLC": max(1, round(prb_frac.get("URLLC", 0.32) * _PRB_TOTAL)),
            "mMTC":  max(1, round(prb_frac.get("mMTC",  0.28) * _PRB_TOTAL)),
        }

        # sla_headroom from last obs vector (obs[12:15]).
        headroom = {}
        if len(self._last_obs) >= 15:
            headroom = {
                "eMBB":  self._last_obs[12],
                "URLLC": self._last_obs[13],
                "mMTC":  self._last_obs[14],
            }

        ep_elapsed  = time.time() - self._ep_start
        run_elapsed = time.time() - self._run_start
        eps_done    = self._ep
        remaining   = max(0, self.total_episodes - eps_done)
        eta_s       = (run_elapsed / max(1, eps_done)) * remaining

        # ── TensorBoard ───────────────────────────────────────────────
        self.writer.add_scalar("reward/episode",         ep_reward,  self._ep)
        self.writer.add_scalar("reward/running_mean10",
                               sum(self._reward_history) / len(self._reward_history),
                               self._ep)
        self.writer.add_scalar("sla/rate",               sla_rate,   self._ep)
        self.writer.add_scalar("throughput/eMBB_Mbps",   embb_thr,   self._ep)
        self.writer.add_scalar("latency/URLLC_ms",       urllc_lat,  self._ep)
        if headroom:
            self.writer.add_scalar("obs/sla_headroom_eMBB",  headroom["eMBB"],  self._ep)
            self.writer.add_scalar("obs/sla_headroom_URLLC", headroom["URLLC"], self._ep)
            self.writer.add_scalar("obs/sla_headroom_mMTC",  headroom["mMTC"],  self._ep)
        if epsilon is not None:
            self.writer.add_scalar("exploration/epsilon",    epsilon,     self._ep)
        if train_steps is not None:
            self.writer.add_scalar("training/train_steps",   train_steps, self._ep)
        if mean_loss is not None and mean_loss > 0.0:
            self.writer.add_scalar("training/mean_loss",     mean_loss,   self._ep)
        if buf_pct is not None:
            self.writer.add_scalar("training/buffer_fill",   buf_pct,     self._ep)
        self.writer.flush()

        # ── Terminal ──────────────────────────────────────────────────
        self._print_episode(ep_elapsed, eta_s / 60.0, headroom)

    # ------------------------------------------------------------------
    # Terminal rendering helpers
    # ------------------------------------------------------------------

    def _prb_bar(self) -> str:
        e = self._prb["eMBB"]
        u = self._prb["URLLC"]
        m = self._prb["mMTC"]
        return (
            f"[red]{'█' * e}[/red][dim] eMBB({e})[/dim]  "
            f"[yellow]{'█' * u}[/yellow][dim] URLLC({u})[/dim]  "
            f"[green]{'█' * m}[/green][dim] mMTC({m})[/dim]"
        )

    def _headroom_bar(self, headroom: dict) -> str:
        """Render sla_headroom as a fraction bar per slice."""
        if not headroom:
            return "[dim]n/a[/dim]"

        def bar(v: float, colour: str) -> str:
            filled = max(0, min(10, round(v * 10)))
            empty  = 10 - filled
            return (
                f"[{colour}]{'▓' * filled}[/{colour}]"
                f"[dim]{'░' * empty}[/dim]"
                f" {v:.2f}"
            )

        return (
            f"[red]eMBB [/red]{bar(headroom.get('eMBB', 0), 'red')}  "
            f"[yellow]URLLC[/yellow]{bar(headroom.get('URLLC', 0), 'yellow')}  "
            f"[green]mMTC [/green]{bar(headroom.get('mMTC', 0), 'green')}"
        )

    def _print_episode(
        self, ep_time: float, eta_min: float, headroom: dict
    ) -> None:
        r_color = "bold green" if self._ep_reward >= 0 else "bold red"
        s_color = (
            "bold green"  if self._sla_rate >= 1.0   else
            "bold yellow" if self._sla_rate >= 0.667 else
            "bold red"
        )

        running_mean = (
            sum(self._reward_history) / len(self._reward_history)
            if self._reward_history else 0.0
        )
        rm_color = "bold green" if running_mean >= 0 else "bold red"

        table = Table(
            title=(
                f"[bold white]{self.agent_name}[/bold white]  "
                f"[dim]ep[/dim] [bold]{self._ep}[/bold]"
                f"[dim]/{self.total_episodes}[/dim]"
            ),
            show_header   = False,
            box           = None,
            padding       = (0, 2),
            title_justify = "left",
        )
        table.add_column(style="bold dim", width=18)
        table.add_column()

        table.add_row("Reward",
            f"[{r_color}]{self._ep_reward:>12.3f}[/{r_color}]")
        table.add_row("Reward mean(10)",
            f"[{rm_color}]{running_mean:>12.3f}[/{rm_color}]")
        table.add_row("SLA rate",
            f"[{s_color}]{self._sla_rate * 100:>6.1f} %[/{s_color}]")
        table.add_row("eMBB thr",   f"{self._embb_thr:>10.2f}  Mbps")
        table.add_row("URLLC lat",  f"{self._urllc_lat:>10.3f}  ms")
        table.add_row("PRB alloc",  self._prb_bar())
        table.add_row("SLA headroom", self._headroom_bar(headroom))

        if self._mean_loss is not None and self._mean_loss > 0.0:
            table.add_row("Mean loss",  f"{self._mean_loss:>12.6f}")
        if self._buf_pct is not None:
            table.add_row("Buffer fill", f"{self._buf_pct * 100:>6.1f} %")

        table.add_row("Ep time",    f"{ep_time:>8.1f}  s")
        table.add_row("ETA",        f"{eta_min:>8.1f}  min")

        if self._epsilon is not None:
            eps_color = "green" if self._epsilon < 0.1 else (
                "yellow" if self._epsilon < 0.5 else "red"
            )
            table.add_row("ε",
                f"[{eps_color}]{self._epsilon:.5f}[/{eps_color}]")
        if self._train_steps is not None:
            table.add_row("Train steps", f"{self._train_steps:,}")

        console.print()
        console.print(table)
        console.print(Rule(style="dim"))
