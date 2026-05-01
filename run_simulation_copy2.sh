#!/usr/bin/env bash
# =============================================================================
# run_simulation.sh
# Launches the NS-3 slice-rl simulation and then starts Python training.
#
# Usage:
#   ./run_simulation.sh [dqn|ppo|r2d2] [--episodes N] [--seed N] [--port N]
#
# Requirements:
#   - NS-3.45 built with 5G-LENA and opengym contrib
#   - Python venv activated OR packages installed globally
#   - NS3_ROOT environment variable set, OR edit NS3_ROOT below
#
# Example:
#   NS3_ROOT=~/ns-3.45 ./run_simulation.sh dqn --episodes 500 --seed 42
# =============================================================================

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
AGENT="${1:-dqn}"
shift || true                          # remaining args forwarded to train script

NS3_ROOT="${NS3_ROOT:-$HOME/5g-project/ns-allinone-3.45/ns-3.45}"
NS3_SCRATCH_BIN="${NS3_ROOT}/cmake-cache/scratch/slice-rl/scratch/slice-rl/ns3.45-slice-rl-sim-default"

GYM_PORT="${GYM_PORT:-5555}"
SIM_SEED="${SIM_SEED:-42}"
#SIM_TIME="${SIM_TIME:-100.0}"
EPISODES=$(echo "$@" | grep -oP '(?<=--episodes )\d+' || echo "500")
SIM_TIME="${SIM_TIME:-$(echo "$EPISODES * 100 + 10" | bc)}"
LOG_DIR="results/logs"
SIM_LOG="${LOG_DIR}/ns3_sim.log"

PYTHON_SCRIPT=""
case "${AGENT}" in
    dqn)  PYTHON_SCRIPT="training/train.py"      ;;
    ppo)  PYTHON_SCRIPT="training/train_ppo.py"  ;;
    r2d2) PYTHON_SCRIPT="training/train_r2d2.py" ;;
    *)
        echo "[ERROR] Unknown agent '${AGENT}'. Choose: dqn | ppo | r2d2"
        exit 1
        ;;
esac

# ── Pre-flight checks ─────────────────────────────────────────────────────────
if [[ ! -f "${NS3_SCRATCH_BIN}" ]]; then
    echo "[ERROR] NS-3 binary not found: ${NS3_SCRATCH_BIN}"
    echo "        Build NS-3 first:"
    echo "        cd ${NS3_ROOT} && ./ns3 build"
    exit 1
fi

if [[ ! -f "${PYTHON_SCRIPT}" ]]; then
    echo "[ERROR] Python script not found: ${PYTHON_SCRIPT}"
    exit 1
fi

if ss -tlnp 2>/dev/null | grep -q ":${GYM_PORT}"; then
    echo "[ERROR] Port ${GYM_PORT} is already in use."
    echo "        Kill the existing process or set: GYM_PORT=5556 ./run_simulation.sh ..."
    exit 1
fi

mkdir -p "${LOG_DIR}" results/models

echo "============================================================"
echo " Agent      : ${AGENT}"
echo " NS-3 bin   : ${NS3_SCRATCH_BIN}"
echo " Gym port   : ${GYM_PORT}"
echo " Seed       : ${SIM_SEED}"
echo " Sim time   : ${SIM_TIME}s"
echo " Sim log    : ${SIM_LOG}"
echo "============================================================"

# ── STEP 1: Start Python FIRST (it is the ZMQ server that NS-3 connects to) ──
#
# ns3-gym architecture:
#   Python Ns3Env  →  ZMQ ROUTER  (binds / listens on GYM_PORT)
#   C++ OpenGymInterface  →  ZMQ DEALER  (connects to Python)
#
# Python must bind before NS-3 launches, otherwise OpenGymInterface's
# connect() call times out immediately and the handshake never completes.
#
echo "[INFO] Starting Python gym server (background): ${PYTHON_SCRIPT} $*"
python "${PYTHON_SCRIPT}" \
    --port  "${GYM_PORT}" \
    --seed  "${SIM_SEED}" \
    "$@" &
PYTHON_PID=$!

# ── STEP 2: Wait until Python has bound the port ─────────────────────────────
echo -n "[INFO] Waiting for Python to open ZMQ port ${GYM_PORT}..."
WAIT_LIMIT=30   # seconds
ELAPSED=0
while ! ss -tlnp 2>/dev/null | grep -q ":${GYM_PORT}"; do
    sleep 0.5
    ELAPSED=$(( ELAPSED + 1 ))
    if (( ELAPSED >= WAIT_LIMIT * 2 )); then
        echo ""
        echo "[ERROR] Python did not open port ${GYM_PORT} within ${WAIT_LIMIT}s."
        echo "        Check for import errors above."
        kill "${PYTHON_PID}" 2>/dev/null || true
        exit 1
    fi
done
echo " ready."

# ── STEP 3: Now launch NS-3 (it will connect to Python's ZMQ socket) ─────────
echo "[INFO] Starting NS-3 simulation (background) — log: ${SIM_LOG}"
"${NS3_SCRATCH_BIN}" \
    --gymPort="${GYM_PORT}" \
    --seed="${SIM_SEED}" \
    --simTime="${SIM_TIME}" \
    > "${SIM_LOG}" 2>&1 &
NS3_PID=$!
echo "[INFO] NS-3 started — PID ${NS3_PID}"

# ── STEP 4: Wait for Python training to complete (it drives the episodes) ─────

echo "[INFO] Training in progress — waiting for Python (PID ${PYTHON_PID}) to finish..." >&2
echo "============================================================" >&2
wait "${PYTHON_PID}"
PYTHON_EXIT=$?
if (( PYTHON_EXIT != 0 )); then
    echo "[WARN] Python exited with code ${PYTHON_EXIT}."
fi

# ── Sanity-check: NS-3 process is alive after 3 s ────────────────────────────
sleep 3
if ! kill -0 "${NS3_PID}" 2>/dev/null; then
    echo "[ERROR] NS-3 process exited unexpectedly within 3 s."
    echo "        Check log: ${SIM_LOG}"
    kill "${PYTHON_PID}" 2>/dev/null || true
    exit 1
fi

# ── Cleanup handler ───────────────────────────────────────────────────────────
cleanup() {
    echo ""
    echo "[INFO] Shutting down NS-3 (PID ${NS3_PID})..."
    kill "${NS3_PID}" 2>/dev/null || true
    wait "${NS3_PID}" 2>/dev/null || true
    echo "[INFO] Done."
}
trap cleanup EXIT INT TERM

