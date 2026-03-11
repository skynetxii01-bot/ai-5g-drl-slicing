"""
envs/slice_gym_env.py
=====================
Python-side OpenGym environment for 5G NR network slicing.
Connects to NS-3 via ns3gym ZMQ interface.

Observation layout (must match C++ GetObservation() EXACTLY):
  obs[0:3]  = PRB fraction per slice    [eMBB, URLLC, mMTC]
  obs[3:6]  = Normalised throughput     [eMBB, URLLC, mMTC]
  obs[6:9]  = Normalised latency        [eMBB, URLLC, mMTC]
  obs[9:12] = Queue occupancy           [eMBB, URLLC, mMTC]
  obs[12:15]= UE count fraction         [eMBB, URLLC, mMTC]

Action space: 27 discrete actions
  action_i = (Δ_eMBB, Δ_URLLC, Δ_mMTC) each Δ ∈ {-1, 0, +1}
"""

import numpy as np
from ns3gym import ns3env

# Slice indices
EMBB  = 0
URLLC = 1
MMTC  = 2
N_SLICES = 3

# SLA targets (3GPP TS 22.261)
SLA = {
    'min_thr_mbps': [10.0, 1.0,  0.1 ],   # [eMBB, URLLC, mMTC]
    'max_lat_ms':   [50.0, 1.0,  500.0],
}

# Normalisation denominators (must match C++ maxThrMbps, maxLatMs)
MAX_THR_MBPS = [100.0, 10.0, 1.0 ]
MAX_LAT_MS   = [50.0,  1.0,  500.0]
MAX_UES      = [10,    5,    20  ]
TOTAL_PRBS   = 25


class SliceGymEnv:
    """
    Wrapper around ns3gym that connects to the NS-3 slice simulation.

    Usage:
        env = SliceGymEnv(port=5555)
        obs = env.reset()
        obs, reward, done, info = env.step(action)
        env.close()
    """

    def __init__(self, port=5555, seed=42):
        self.port = port
        self.seed = seed

        # Connect to NS-3 via ns3gym
        # NS-3 must already be running and listening on this port!
        self.env = ns3env.Ns3Env(
            port=port,
            stepTime=0.1,      # 100ms gym step
            startSim=0,        # NS-3 is already started externally
            simSeed=seed,
            simArgs={},
            debug=False
        )

        self.observation_space = self.env.observation_space
        self.action_space      = self.env.action_space

        # Track current PRB allocation for baseline policies
        self._prb_alloc = np.array([13, 7, 5], dtype=np.int32)
        self._step_count = 0
        self._rr_pointer = 0   # Round-robin pointer

    def reset(self):
        """Reset environment to start of new episode."""
        obs = self.env.reset()
        self._step_count = 0
        self._prb_alloc  = np.array([13, 7, 5], dtype=np.int32)
        return self._process_obs(obs)

    def step(self, action):
        """
        Apply action and return (obs, reward, done, info).
        action: int in [0, 26]
        """
        obs, reward, done, info = self.env.step(action)
        self._step_count += 1

        # Update local PRB tracking from observation
        processed_obs = self._process_obs(obs)
        if processed_obs is not None:
            self._prb_alloc = (processed_obs[0:3] * TOTAL_PRBS).astype(np.int32)
            self._prb_alloc = np.maximum(self._prb_alloc, 1)

        return processed_obs, reward, done, info

    def close(self):
        """Close the connection to NS-3."""
        self.env.close()

    # ─────────────────────────────────────────────────────────────────────────
    # Observation processing
    # ─────────────────────────────────────────────────────────────────────────

    def _process_obs(self, obs):
        """
        Process raw observation from NS-3 into numpy array.
        Validates shape and clips to [0, 1].
        """
        if obs is None:
            return np.zeros(15, dtype=np.float32)

        obs = np.array(obs, dtype=np.float32).flatten()
        if len(obs) != 15:
            # Pad or truncate if NS-3 sends wrong size
            padded = np.zeros(15, dtype=np.float32)
            padded[:min(len(obs), 15)] = obs[:15]
            obs = padded

        return np.clip(obs, 0.0, 1.0)

    def decode_obs(self, obs):
        """
        Decode observation vector into named components.
        Useful for logging and debugging.
        """
        return {
            'prb_fraction': obs[0:3],          # [eMBB, URLLC, mMTC]
            'throughput':   obs[3:6],           # normalised [0,1]
            'latency':      obs[6:9],           # normalised [0,1]
            'queue':        obs[9:12],          # occupancy [0,1]
            'ue_fraction':  obs[12:15],         # UE count fraction
            # Denormalised values
            'thr_mbps':     obs[3:6] * MAX_THR_MBPS,
            'lat_ms':       obs[6:9] * MAX_LAT_MS,
            'prbs':         (obs[0:3] * TOTAL_PRBS).astype(int),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Baseline policies (for comparison with DRL agents)
    # ─────────────────────────────────────────────────────────────────────────

    def random_policy(self, obs):
        """Random action — baseline lower bound."""
        return self.action_space.sample()

    def round_robin_policy(self, obs):
        """
        Round-robin: cycle through +1 to each slice in turn.
        Actions: give 1 PRB to next slice, take from largest.
        """
        # Action that increments the current slice pointer by 1
        # Δ vectors: [+1,0,0]=18, [0,+1,0]=14 (approx), [0,0,+1]=10 (approx)
        rr_actions = [
            self._encode_action(+1, -1, 0),   # give to eMBB
            self._encode_action(-1, +1, 0),   # give to URLLC
            self._encode_action(0,  -1, +1),  # give to mMTC
        ]
        action = rr_actions[self._rr_pointer % N_SLICES]
        self._rr_pointer += 1
        return action

    def proportional_fair_policy(self, obs):
        """
        Proportional Fair: give more PRBs to slices with low throughput
        relative to their SLA minimum.
        """
        thr_mbps = obs[3:6] * np.array(MAX_THR_MBPS)
        min_thr  = np.array(SLA['min_thr_mbps'])

        # PF metric: deficit from SLA target
        deficit = np.maximum(min_thr - thr_mbps, 0)

        if deficit.sum() < 1e-6:
            # All slices meeting SLA — no change needed
            return self._encode_action(0, 0, 0)

        # Give 1 PRB to the slice with highest deficit
        worst_slice = np.argmax(deficit)
        best_slice  = np.argmin(self._prb_alloc)

        deltas = [0, 0, 0]
        if worst_slice != best_slice:
            deltas[worst_slice] = +1
            deltas[np.argmax(self._prb_alloc)] = -1

        return self._encode_action(*deltas)

    def _encode_action(self, d_embb, d_urllc, d_mmtc):
        """
        Encode (Δ_eMBB, Δ_URLLC, Δ_mMTC) into action index [0, 26].
        Each Δ ∈ {-1, 0, +1} → digit {0, 1, 2}
        """
        def to_digit(d): return int(d) + 1
        return to_digit(d_embb) * 9 + to_digit(d_urllc) * 3 + to_digit(d_mmtc)

    # ─────────────────────────────────────────────────────────────────────────
    # SLA metrics
    # ─────────────────────────────────────────────────────────────────────────

    def compute_sla_compliance(self, obs):
        """
        Returns fraction of SLA targets met (0.0 to 1.0).
        Each slice has 2 targets: throughput and latency.
        """
        thr_mbps = obs[3:6] * np.array(MAX_THR_MBPS)
        lat_ms   = obs[6:9] * np.array(MAX_LAT_MS)

        min_thr  = np.array(SLA['min_thr_mbps'])
        max_lat  = np.array(SLA['max_lat_ms'])

        thr_ok = (thr_mbps >= min_thr)
        lat_ok = (lat_ms   <= max_lat)

        return (thr_ok.sum() + lat_ok.sum()) / (2 * N_SLICES)
