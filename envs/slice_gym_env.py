"""Gym 0.26.2 wrapper for the NS-3 5G slice RL environment.

Observation layout (15 floats, all normalised to [0, 1]):
    [0:3]   prb_frac      — fraction of totalPrbs allocated to each slice
    [3:6]   throughput    — normalised by per-slice maxThrMbps
    [6:9]   latency       — normalised by 2 × per-slice maxLatMs
                            (reads 0.5 at the SLA boundary, not 1.0)
    [9:12]  hol_delay     — mean Head-of-Line delay per slice,
                            normalised by 2 × per-slice maxLatMs.
                            SLA boundary maps to 0.5, matching obs[6:9].
                            Forward-looking congestion proxy: rises before
                            packets are dropped. Sample-and-hold when no
                            fresh scheduler callback fires in the 100ms window.
    [12:15] sla_headroom  — (thr - minThr) / (maxThr - minThr), clipped to [0, 1].
                            0.0 = at or below SLA floor.
                            0.5 = halfway between floor and capacity.
                            1.0 = at capacity ceiling.
                            Provides gradient across the full realistic operating
                            range. Replaces load_pressure (thr/minThr) which
                            saturated to 1.0 at normal loads.
    [15:18] demand_active — 1.0 if slice had fresh measurements this step,
                            0.0 if in off-period. Disambiguates stale
                            sample-and-hold latency from fresh measurements.      
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple
import json 

import gym
import numpy as np
from gym import spaces
import json

# ns3gym is installed inside the project venv only.
# The guarded import below:
#   (a) silences IDE warnings from static analysers that don't know the venv
#   (b) produces a clear, actionable error if someone runs this outside the venv
# At runtime inside the venv this import always succeeds — it is not a real guard.
try:
    from ns3gym import ns3env
except ImportError as e:
    raise ImportError(
        "ns3gym is not installed in the active Python environment.\n"
        "Activate the project venv first:\n"
        "  source ~/5g-project/ns-allinone-3.45/.venv/bin/activate\n"
        f"Original error: {e}"
    ) from e


SLICE_NAMES = ("eMBB", "URLLC", "mMTC")
OBS_SIZE = 18 
ACTION_SIZE = 27


class SliceGymEnv(gym.Env):
    """Gym 0.26-compatible wrapper over ns3-gym for 3-slice PRB control."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        port: int = 5555,
        sim_seed: int = 42,
        debug: bool = False,
        start_sim: bool = False,
        sim_args: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()

        self.port = int(port)
        self.sim_seed = int(sim_seed)
        self.debug = bool(debug)
        self.start_sim = bool(start_sim)
        self.sim_args = sim_args or {}

        self.observation_space = spaces.Box(
            low=np.float32(0.0),
            high=np.float32(1.0),
            shape=(OBS_SIZE,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(ACTION_SIZE)

        # Ns3Env is the ZMQ bridge object. It binds to GYM_PORT and waits
        # for the NS-3 process (which acts as the ZMQ client) to connect.
        # startSim=False means we never launch NS-3 from Python —
        # run_simulation_copy2.sh handles that separately.
        self._ns3_env = ns3env.Ns3Env(
            port=self.port,
            simSeed=self.sim_seed,
            simArgs=self.sim_args,
            startSim=self.start_sim,
            debug=self.debug,
        )

    def _validate_obs(self, obs: np.ndarray) -> np.ndarray:
        arr = np.asarray(obs, dtype=np.float32).reshape(-1)
        if arr.shape[0] != OBS_SIZE:
            raise ValueError(
                f"Expected EXACT {OBS_SIZE}-dim observation from NS-3, got {arr.shape[0]}"
            )
        return arr
    
    def _coerce_obs(self, obs_obj: Any) -> np.ndarray:
        """Unwrap ns3-gym observation payload variants into a flat 15-vector."""
        obj = obs_obj
        # Some ns3-gym builds wrap observation in singleton tuple/list.
        for _ in range(3):
            if isinstance(obj, tuple) and len(obj) == 1:
                obj = obj[0]
                continue
            if isinstance(obj, list) and len(obj) == 1 and isinstance(obj[0], (list, tuple, np.ndarray)):
                obj = obj[0]
                continue
            break
        # Some wrappers return dict-like payloads for reset/step.
        if isinstance(obj, dict):
            for key in ("obs", "observation", "state", "data"):
                if key in obj:
                    obj = obj[key]
                    break
        # Some wrappers return textual vectors like "[0.1, 0.2, ...]".
        if isinstance(obj, str):
            s = obj.strip()
            if s.startswith("[") and s.endswith("]"):
                try:
                    parsed = json.loads(s)
                    obj = parsed
                except Exception:
                    pass
        # Last-chance unwrap after dict/string conversion.
        if isinstance(obj, (tuple, list)) and len(obj) == 1 and isinstance(obj[0], (list, tuple, np.ndarray)):
            obj = obj[0]
        arr = np.asarray(obj, dtype=np.float32).reshape(-1)
        if arr.shape[0] == OBS_SIZE:
            return arr
        raise ValueError(
            "Expected EXACT 18-dim observation from NS-3, got "
            f"{arr.shape[0]} | type={type(obs_obj).__name__} | repr={repr(obs_obj)[:220]}"
        )

    def _decode_obs(self, obs: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Break the flat 15-float observation into a labelled dictionary.

        IMPORTANT: the third channel (obs[9:12]) is labelled 'hol_delay'.
        It was renamed from 'queue_occ' in session 2026-05-03 to match the
        C++ implementation in slice-env.cc (AggregateHolDelay).
        Do NOT revert this key name.
        """
        arr = self._validate_obs(obs)
        return {
            "prb_frac":   dict(zip(SLICE_NAMES, arr[0:3].tolist())),
            "throughput": dict(zip(SLICE_NAMES, arr[3:6].tolist())),
            "latency":    dict(zip(SLICE_NAMES, arr[6:9].tolist())),
            "hol_delay":  dict(zip(SLICE_NAMES, arr[9:12].tolist())),  # was queue_occ — FINAL
            "sla_headroom" :  dict ( zip (SLICE_NAMES, arr[ 12 : 15 ].tolist())),
            "demand_active": dict(zip(SLICE_NAMES, arr[15:18].tolist())),
        }
    
    @staticmethod
    def _coerce_info(info_obj: Any) -> Dict[str, Any]:
        """Normalize backend info payloads to a dictionary.

        ns3-gym versions are inconsistent: `info` may be a dict, mapping-like
        object, JSON string, plain string, or None.
        """
        if info_obj is None:
            return {}
        if isinstance(info_obj, dict):
            return dict(info_obj)
        if isinstance(info_obj, Mapping):
            return dict(info_obj.items())
        if isinstance(info_obj, str):
            s = info_obj.strip()
            if s.startswith("{"):
                try:
                    parsed = json.loads(s)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    pass
            return {"extraInfo": info_obj}
        return {"extraInfo": str(info_obj)}


    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        del options
        if seed is not None:
            self.sim_seed = int(seed)

        raw = self._ns3_env.reset()

        # ns3-gym wrappers differ by version: old returns obs, newer may return (obs, info)
        if isinstance(raw, tuple) and len(raw) == 2:
            obs_raw, info = raw
            info = self._coerce_info(info)
        else:
            obs_raw, info = raw, {}

        obs = self._coerce_obs(obs_raw)
        info["decoded_obs"] = self._decode_obs(obs)
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        act = int(action)
        if act < 0 or act >= ACTION_SIZE:
            raise ValueError(f"Action must be in [0, {ACTION_SIZE - 1}], got {act}")

        raw = self._ns3_env.step(act)

        # Handle both old (obs, reward, done, info) and 0.26-like 5-value returns.
        if isinstance(raw, tuple) and len(raw) == 5:
            obs_raw, reward, done, truncated, info = raw
        elif isinstance(raw, tuple) and len(raw) == 4:
            obs_raw, reward, done, info = raw
            truncated = False
        else:
            raise RuntimeError("Unexpected step() return format from ns3-gym backend")

        obs = self._validate_obs(obs_raw)
        info = self._coerce_info(info)
        extra = info.get("extraInfo")
        if isinstance(extra, str) and extra.strip().startswith("{"):
            try:
                info["extra_json"] = json.loads(extra)
            except Exception:
                pass
        if "extra_json" not in info and not hasattr(self, "_extrainfo_warned"):
            self._extrainfo_warned = True
            print(
                "[WARN] SliceGymEnv: extraInfo did not parse to JSON on first step. "
                "demand_active will default to all-active — SLA rate may be inflated. "
                "Check ns3-gym version and NrSliceGymEnv::GetExtraInfo() output."
            )

        info["decoded_obs"] = self._decode_obs(obs)

        return obs, float(reward), bool(done), bool(truncated), info

    def render(self) -> None:
        return None

    def close(self) -> None:
        if self._ns3_env is not None:
            self._ns3_env.close()


__all__ = ["SliceGymEnv", "SLICE_NAMES", "OBS_SIZE", "ACTION_SIZE"]
