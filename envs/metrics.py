"""envs/metrics.py
Shared metric utilities for training and evaluation scripts.

All functions here are agent-agnostic — they operate on environment
outputs (decoded_obs, extra_json, cfg) only. Import from here in every
training and evaluation script to avoid definition drift.

    from envs.metrics import compute_sla_rates, nan_or_round
"""

from __future__ import annotations

from typing import Dict, List, Any

from envs.slice_gym_env import SLICE_NAMES


def compute_sla_rates(
    decoded_obs: Dict,
    cfg: Dict,
    extra_json: Dict | None = None,
) -> Dict[str, float]:
    """Compute SLA satisfaction rates — overall and per slice.

    Parameters
    ----------
    decoded_obs : info["decoded_obs"] from env.step()
    cfg         : loaded configs/config.yaml dict
    extra_json  : info["extra_json"] from env.step() — used for
                  demand_active flags. Falls back to all-active if absent.

    Returns
    -------
    dict with keys:
        "overall"  — fraction of active slices meeting SLA  (float in [0,1])
        "embb"     — 1.0 satisfied / 0.0 violated  (inactive counts as 1.0)
        "urllc"    — same
        "mmtc"     — same

    Rules
    -----
    - Inactive slices (demand_active == 0) are excluded from the overall
      denominator and reported as 1.0 individually (not a violation).
    - Latency SLA boundary: lat_norm <= 0.5
      (obs[6:9] is normalised by 2*maxLatMs, so SLA sits at 0.5 not 1.0).
    - Throughput SLA boundary: thr >= min_thr_mbps (denormalised from obs).

    Must remain consistent with:
        C++  slice-env.cc  ScheduleStep()  slaSat computation
        train.py           compute_sla_rates (now removed — import from here)
        evaluate.py        compute_sla_rates (now removed — import from here)
        train_ppo.py       (import from here)
        train_r2d2.py      (import from here)
    """
    max_thr = cfg["env"]["max_thr_mbps"]
    min_thr = cfg["env"]["min_thr_mbps"]

    demand_active = [1, 1, 1]
    if (
        extra_json
        and isinstance(extra_json.get("demand_active"), list)
        and len(extra_json["demand_active"]) == 3
    ):
        demand_active = [int(x) for x in extra_json["demand_active"]]

    per_slice: List[float] = []
    sat = 0
    den = 0

    for i, s in enumerate(SLICE_NAMES):
        if demand_active[i] == 0:
            per_slice.append(1.0)   # inactive — not a violation
            continue
        den += 1
        thr      = float(decoded_obs["throughput"][s]) * float(max_thr[s])
        lat_norm = float(decoded_obs["latency"][s])
        ok       = thr >= float(min_thr[s]) and lat_norm <= 0.5
        per_slice.append(1.0 if ok else 0.0)
        if ok:
            sat += 1

    return {
        "overall": 1.0 if den == 0 else sat / den,
        "embb":    per_slice[0],
        "urllc":   per_slice[1],
        "mmtc":    per_slice[2],
    }


def nan_or_round(v: float, decimals: int = 6) -> float | None:
    """Round a float for JSON serialisation; return None for NaN."""
    if isinstance(v, float) and v != v:   # NaN check (NaN != NaN)
        return None
    return round(float(v), decimals)
