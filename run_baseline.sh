#!/usr/bin/env bash
set -euo pipefail

NS3_ROOT="${NS3_ROOT:-$HOME/5g-project/ns-allinone-3.45/ns-3.45}"
EPISODES="${EPISODES:-10}"
MAX_STEPS="${MAX_STEPS:-1000}"
SEED="${SEED:-42}"
OUT="${OUT:-results/logs/baseline_cpp.json}"

cd "$NS3_ROOT"

echo "[INFO] Building baseline target..."
cmake --build cmake-cache --target scratch_slice-rl_slice-rl-sim-baseline -j"$(nproc)"

BIN="${NS3_ROOT}/cmake-cache/scratch/slice-rl/scratch/slice-rl/ns3.45-slice-rl-sim-baseline-default"
if [[ ! -x "$BIN" ]]; then
  echo "[ERROR] Baseline binary not found: $BIN"
  exit 1
fi

echo "[INFO] Running baseline..."
"$BIN" --episodes="$EPISODES" --maxSteps="$MAX_STEPS" --seed="$SEED" --out="$OUT"

echo "[INFO] Done. Output: $NS3_ROOT/$OUT"
