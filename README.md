# AI-Powered Dynamic Resource Allocation in 5G NR Network Slicing Using Deep Reinforcement Learning

This repository provides a complete NS-3 + 5G-LENA + ns3-gym + PyTorch research stack for dynamic PRB allocation across 3 network slices (eMBB/URLLC/mMTC).

## Architecture (ASCII)

```text
+---------------------+       ZMQ/OpenGym        +-----------------------------+
| NS-3.45 + 5G-LENA   | <----------------------> | Python RL Runtime           |
| slice-rl-sim.cc     |                          | SliceGymEnv + DQN/PPO/R2D2 |
| NrMacSchedulerTdmaAi|                          | training/evaluate scripts   |
+----------+----------+                          +--------------+--------------+
           |                                                        |
           v                                                        v
   Throughput/Latency/SLA Metrics                          Models + Logs + Figures
```

## Prerequisites

- NS-3.45 installed (CMake `./ns3` build workflow).
- 5G-LENA NR v4.1.y already present in `contrib/nr/`.
- ns3-gym already present in `contrib/opengym/`.

## Installation

```bash
pip3 install --break-system-packages torch numpy matplotlib scipy pyyaml
pip3 install --break-system-packages gym==0.21.0   # ALWAYS LAST
pip3 install --break-system-packages contrib/opengym/model/ns3gym
```

## Run (TWO TERMINALS REQUIRED)

Terminal 1 (start NS-3 first so it binds ZMQ port):

```bash
./ns3 run "scratch/slice-rl/slice-rl-sim --gymPort=5555 --seed=42"
```

Terminal 2 (training process):

```bash
python3 training/train.py --agent dqn --port 5555 --seed 42
```

Other agents:

```bash
python3 training/train.py --agent ppo --port 5555 --seed 42
python3 training/train.py --agent r2d2 --port 5555 --seed 42
```

Evaluation:

```bash
python3 training/evaluate.py --port 5555 --seed 99 --agent dqn
```

Plotting:

```bash
python3 analysis/plot_results.py
```

## Expected Results (example trend)

| Policy | Reward ↑ | eMBB Thr ↑ | URLLC Lat ↓ | SLA % ↑ |
|---|---:|---:|---:|---:|
| Random | Low | Low | High | Low |
| Round Robin | Medium-Low | Medium-Low | Medium-High | Medium-Low |
| Proportional Fair | Medium | Medium | Medium | Medium |
| DQN | High | High | Medium-Low | High |
| PPO | High | High | Medium-Low | High |
| R2D2 | Highest | Highest | Lowest | Highest |

## Folder Structure

```text
.
├── agents/
│   ├── dqn/
│   ├── ppo/
│   └── r2d2/
├── analysis/
│   └── plot_results.py
├── configs/
│   └── config.yaml
├── envs/
│   └── slice_gym_env.py
├── ns3-workspace/
│   └── scratch/slice-rl/
│       ├── CMakeLists.txt
│       ├── slice-env.h
│       ├── slice-env.cc
│       └── slice-rl-sim.cc
├── training/
│   ├── train.py
│   └── evaluate.py
├── requirements.txt
└── .gitignore
```

## Notes

- Uses CMake/`./ns3` only (no waf/wscript).
- Gym version is pinned to `0.21.0` for ns3-gym compatibility.
- NS-3 process must start before Python client.
