"""
training/monitor.py
Unified training monitor for DQN / PPO / R2D2.

Provides:
  - tqdm step-level progress bar  (x / max_steps per episode)
  - Rich per-episode summary table with PRB allocation bar
  - TensorBoard scalar logging
  - Wall-clock timing and ETA

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
                # ... store / train ...
                mon.step()
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
                # train_steps = agent.train_steps  # R2D2 only
            )
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional

from rich.console import Console
from rich.rule import Rule
from rich.table import Table
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

console = Console()

_PRB_TOTAL = 25          # must match slice-rl-sim.cc totalPrbs


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
        self._epsilon     : Optional[float]  = None
        self._train_steps : Optional[int]    = None
        self._prb         : Dict[str, int]   = {"eMBB": 10, "URLLC": 8, "mMTC": 7}

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "TrainingMonitor":
        self._run_start = time.time()
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
        self._step_bar = tqdm(
            total         = self.max_steps,
            desc          = f"[{self.agent_name}] ep {ep:>4}/{self.total_episodes}",
            unit          = "step",
            leave         = False,
            dynamic_ncols = True,
            colour        = "cyan",
        )
        return self._step_bar

    def step(
        self,
        ep_reward : float        = 0.0,
        epsilon   : float | None = None,
        loss      : float | None = None,
        buf_pct   : float | None = None,
        sla_rate  : float | None = None,
    ) -> None:
        """Advance the tqdm bar by one step. Call once per env.step()."""
        if self._step_bar is not None:
            pf: dict = {"r": f"{ep_reward:.1f}"}
            if epsilon  is not None: pf["ε"]    = f"{epsilon:.3f}"
            if loss     is not None: pf["loss"] = f"{loss:.4f}"
            if buf_pct  is not None: pf["buf"]  = f"{buf_pct:.0%}"
            if sla_rate is not None: pf["sla"]  = f"{sla_rate:.0%}"
            self._step_bar.set_postfix(pf)
            self._step_bar.update(1)

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
    ) -> None:
        """
        Call AFTER the step loop, before logging to JSONL.

        decoded_obs : info["decoded_obs"] from the last env.step()
        epsilon     : agent.epsilon()      — DQN only, omit for PPO/R2D2
        train_steps : agent.train_steps    — R2D2 only, omit for DQN/PPO
        """
        self._ep_reward   = ep_reward
        self._sla_rate    = sla_rate
        self._embb_thr    = embb_thr
        self._urllc_lat   = urllc_lat
        self._epsilon     = epsilon
        self._train_steps = train_steps
        self._mean_loss   = mean_loss

        # Decode PRB allocation from normalised observation
        prb_frac = (decoded_obs or {}).get("prb_frac", {})
        self._prb = {
            "eMBB":  max(1, round(prb_frac.get("eMBB",  0.40) * _PRB_TOTAL)),
            "URLLC": max(1, round(prb_frac.get("URLLC", 0.32) * _PRB_TOTAL)),
            "mMTC":  max(1, round(prb_frac.get("mMTC",  0.28) * _PRB_TOTAL)),
        }

        ep_elapsed  = time.time() - self._ep_start
        run_elapsed = time.time() - self._run_start
        eps_done    = self._ep  # episodes done including this one
        remaining   = max(0, self.total_episodes - eps_done)
        eta_s       = (run_elapsed / max(1, eps_done)) * remaining

        # ── TensorBoard ───────────────────────────────────────────────
        self.writer.add_scalar("reward/episode",        ep_reward,  self._ep)
        self.writer.add_scalar("sla/rate",              sla_rate,   self._ep)
        self.writer.add_scalar("throughput/eMBB_Mbps",  embb_thr,   self._ep)
        self.writer.add_scalar("latency/URLLC_ms",      urllc_lat,  self._ep)
        if epsilon is not None:
            self.writer.add_scalar("exploration/epsilon",   epsilon,     self._ep)
        if train_steps is not None:
            self.writer.add_scalar("training/train_steps",  train_steps, self._ep)
        if mean_loss is not None and mean_loss > 0.0:
            self.writer.add_scalar("training/mean_loss",    mean_loss,   self._ep)
        self.writer.flush()

        # ── Terminal ──────────────────────────────────────────────────
        self._print_episode(ep_elapsed, eta_s / 60.0)

    # ------------------------------------------------------------------
    # Terminal rendering helpers
    # ------------------------------------------------------------------

    def _prb_bar(self) -> str:
        e = self._prb["eMBB"]
        u = self._prb["URLLC"]
        m = self._prb["mMTC"]
        return (
            f"[cyan]{'█' * e}[/cyan][dim] eMBB({e})[/dim]  "
            f"[yellow]{'█' * u}[/yellow][dim] URLLC({u})[/dim]  "
            f"[green]{'█' * m}[/green][dim] mMTC({m})[/dim]"
        )

    def _print_episode(self, ep_time: float, eta_min: float) -> None:
        r_color = "bold green" if self._ep_reward >= 0 else "bold red"
        s_color = (
            "bold green"  if self._sla_rate >= 1.0   else
            "bold yellow" if self._sla_rate >= 0.667 else
            "bold red"
        )

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
        table.add_column(style="bold dim", width=16)
        table.add_column()

        table.add_row("Reward",
            f"[{r_color}]{self._ep_reward:>12.3f}[/{r_color}]")
        table.add_row("SLA rate",
            f"[{s_color}]{self._sla_rate * 100:>6.1f} %[/{s_color}]")
        table.add_row("eMBB thr",  f"{self._embb_thr:>10.2f}  Mbps")
        table.add_row("URLLC lat", f"{self._urllc_lat:>10.3f}  ms")
        table.add_row("PRB alloc", self._prb_bar())
        table.add_row("Ep time",   f"{ep_time:>8.1f}  s")
        table.add_row("ETA",       f"{eta_min:>8.1f}  min")
        if self._epsilon is not None:
            table.add_row("ε",     f"{self._epsilon:.5f}")
        if self._train_steps is not None:
            table.add_row("Steps", f"{self._train_steps:,}")

        console.print()
        console.print(table)
        console.print(Rule(style="dim"))