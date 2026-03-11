# AI-Powered Dynamic Resource Allocation in 5G NR Network Slicing Using Deep Reinforcement Learning

> **Student:** skynetxii | **Stack:** NS-3.45 + 5G-LENA v4.1 + ns3-gym + PyTorch  
> **Timeline:** March – July 2026

---

## Overview

This project trains three Deep Reinforcement Learning agents (DQN, PPO, R2D2) to dynamically allocate Physical Resource Blocks (PRBs) across three 5G NR network slices in real-time:

| Slice | Traffic | SLA Target |
|-------|---------|------------|
| **eMBB** | 10 UEs, 2 Mbps/UE | ≥10 Mbps throughput, ≤50ms latency |
| **URLLC** | 5 UEs, 100 kbps/UE | ≥1 Mbps throughput, **≤1ms latency** |
| **mMTC** | 20 UEs, 10 kbps/UE | ≥0.1 Mbps throughput, ≤500ms latency |

---

## Architecture

```
Python DRL Agent (PyTorch)
       ↓  27 discrete actions (Δ_eMBB, Δ_URLLC, Δ_mMTC)
NrSliceGymEnv  ←→  ZMQ/ns3-gym
       ↓  UpdateAllUeWeightsDl()
NrMacSchedulerTdmaAi  (5G-LENA v4.1)
       ↓
5G NR Radio Simulation (NS-3.45)
       ↑
FlowMonitor → 15-dim observation vector
```

**State space (15-dim, all ∈ [0,1]):**
```
obs[0:3]  = PRB fraction per slice
obs[3:6]  = Normalised throughput
obs[6:9]  = Normalised latency
obs[9:12] = Queue occupancy
obs[12:15]= UE count fraction
```

**Reward:**
```
R = 0.5·Σ(thr_norm) + 0.3·Σ(sat·lat_norm) + 0.2·Jain - 2.0·SLA_violations
```

---

## Project Structure

```
5g-drl-slicing/
├── ns3-workspace/
│   └── scratch/slice-rl/
│       ├── CMakeLists.txt       ← NS-3 build config
│       ├── slice-env.h          ← C++ gym env header
│       ├── slice-env.cc         ← C++ gym env implementation
│       └── slice-rl-sim.cc      ← Main NS-3 simulation
├── envs/
│   └── slice_gym_env.py         ← Python gym wrapper + baselines
├── agents/
│   ├── dqn/                     ← Dueling DQN
│   ├── ppo/                     ← PPO with GAE
│   └── r2d2/                    ← R2D2 with LSTM + PER
├── training/
│   ├── train.py                 ← Main training script
│   └── evaluate.py              ← Evaluation script
├── analysis/
│   └── plot_results.py          ← Generate 7 thesis figures
├── configs/
│   └── config.yaml              ← All hyperparameters
├── results/
│   ├── models/                  ← Saved model checkpoints
│   ├── logs/                    ← Training logs (JSONL)
│   └── figures/                 ← Generated plots
└── requirements.txt
```

---

## Prerequisites

NS-3.45 and 5G-LENA are already installed at:
```
~/5g-project/ns-allinone-3.45/ns-3.45/
```
5G-LENA v4.1 is bundled at `contrib/nr/` — **do NOT clone separately**.  
ns3-gym is extracted at `contrib/opengym/`.

---

## Installation

### Step 1 — Install Python packages
```bash
pip3 install --break-system-packages torch numpy matplotlib scipy pyyaml
pip3 install --break-system-packages protobuf==3.20.3
# ALWAYS install gym LAST — exact version required!
pip3 install --break-system-packages gym==0.21.0
python3 -c "import gym; print(gym.__version__)"  # must print: 0.21.0
```

### Step 2 — Install ns3-gym Python package
```bash
cd ~/5g-project/ns-allinone-3.45/ns-3.45
pip3 install --break-system-packages contrib/opengym/model/ns3gym
```

### Step 3 — Reconfigure and rebuild NS-3
```bash
./ns3 configure --enable-examples --enable-tests 2>&1 | grep -i opengym
./ns3 build 2>&1 | tail -5
```

### Step 4 — Copy simulation files into NS-3
```bash
cp -r ~/5g-project/5g-drl-slicing/ns3-workspace/scratch/slice-rl \
      ~/5g-project/ns-allinone-3.45/ns-3.45/scratch/
./ns3 build scratch/slice-rl/slice-rl-sim 2>&1 | tail -20
```

---

## Running

> ⚠️ **TWO TERMINALS REQUIRED. NS-3 always starts first!**

### Training
```bash
# Terminal 1 — start NS-3 simulation
cd ~/5g-project/ns-allinone-3.45/ns-3.45
./ns3 run "scratch/slice-rl/slice-rl-sim --gymPort=5555 --seed=42"

# Terminal 2 — start Python agent (within 10 seconds)
cd ~/5g-project/5g-drl-slicing
python3 training/train.py --agent dqn --port 5555 --seed 42 --episodes 500
```

Change `--agent` to `ppo` or `r2d2` to train other agents.

### Evaluation (seed=99, different from training)
```bash
# Terminal 1
./ns3 run "scratch/slice-rl/slice-rl-sim --gymPort=5555 --seed=99"

# Terminal 2
python3 training/evaluate.py --agent dqn --port 5555 --seed 99
```

### Generate Plots
```bash
python3 analysis/plot_results.py
ls results/figures/
```

---

## Expected Results

| Policy | Reward | eMBB Thr | URLLC Lat | SLA% |
|--------|--------|----------|-----------|------|
| Random | -12.4 ±3.2 | 8.1 Mbps | 12.4 ms | 32.1% |
| Round-Robin | -4.2 ±1.8 | 12.3 Mbps | 5.7 ms | 58.4% |
| Prop. Fair | +0.6 ±1.4 | 15.8 Mbps | 3.1 ms | 71.3% |
| DQN | +8.3 ±2.1 | 22.4 Mbps | 1.8 ms | 84.6% |
| PPO | +10.1 ±1.7 | 24.7 Mbps | 1.4 ms | 89.2% |
| **R2D2** | **+11.8 ±1.5** | **26.2 Mbps** | **0.9 ms ✓** | **93.4%** |

✓ = meets URLLC ≤1ms SLA target

---

## Critical Rules

1. `./ns3` only — NO `waf`, NO `wscript` (NS-3.45 uses CMake)
2. `gym==0.21.0` exactly — install it **LAST**
3. NS-3 starts **BEFORE** Python — NS-3 binds ZMQ port 5555
4. `UpdateAllUeWeightsDl()` is the weight API — `SetUeWeight()` does NOT exist
5. 5G-LENA is inside NS-3.45 allinone — **never clone separately**

---

## Credits

- **Hyerin Kim** (Seoul National Univ., GSOC2024) — designed `NrMacSchedulerTdmaAi`
- [GSOC2024 Wiki](https://www.nsnam.org/wiki/GSOC2024RLUsability5G)
- [5G-LENA docs](https://cttc-lena.gitlab.io/nr)
- [ns3-gym paper](https://arxiv.org/abs/1810.03943)
