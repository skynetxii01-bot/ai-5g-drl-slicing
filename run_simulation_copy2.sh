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
shift || true   # remaining args forwarded to Python train script

NS3_ROOT="${NS3_ROOT:-$HOME/5g-project/ns-allinone-3.45/ns-3.45}"
NS3_SCRATCH_BIN="${NS3_ROOT}/cmake-cache/scratch/slice-rl/scratch/slice-rl/ns3.45-slice-rl-sim-default"

GYM_PORT="${GYM_PORT:-5555}"
SIM_SEED="${SIM_SEED:-42}"
LOG_DIR="results/logs"
SIM_LOG="${LOG_DIR}/ns3_sim.log"

# Parse --episodes from forwarded args safely.
# Default: 500. grep returns exit 1 if not found; || echo handles that.
EPISODES=500
for arg in "$@"; do
    if [[ "$arg" =~ ^[0-9]+$ ]] && [[ "${prev_arg:-}" == "--episodes" ]]; then
        EPISODES="$arg"
    fi
    prev_arg="$arg"
done

# sim_time = episodes × 100 s (1000 steps × 100 ms) + 10 s buffer
SIM_TIME="${SIM_TIME:-$(echo "$EPISODES * 100 + 10" | bc)}"

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

# ── Cleanup handler — registered BEFORE any background processes start ────────
# This ensures NS-3 is always killed if the script exits for any reason,
# including Python crashes, SIGINT (Ctrl-C), and SIGTERM.
NS3_PID=""
TAIL_PID=""
cleanup() {
    echo ""
    echo "[INFO] Cleanup: shutting down background processes..."
    if [[ -n "${NS3_PID}" ]] && kill -0 "${NS3_PID}" 2>/dev/null; then
        echo "[INFO]   Killing NS-3 (PID ${NS3_PID})..."
        kill "${NS3_PID}" 2>/dev/null || true
        wait "${NS3_PID}" 2>/dev/null || true
    fi
    if [[ -n "${TAIL_PID}" ]] && kill -0 "${TAIL_PID}" 2>/dev/null; then
        kill "${TAIL_PID}" 2>/dev/null || true
    fi
    echo "[INFO] Done."
}
trap cleanup EXIT INT TERM

# ── Pre-flight checks ─────────────────────────────────────────────────────────
if [[ ! -f "${NS3_SCRATCH_BIN}" ]]; then
    echo "[ERROR] NS-3 binary not found: ${NS3_SCRATCH_BIN}"
    echo "        Build NS-3 first:"
    echo "        cd ${NS3_ROOT} && cmake --build build/default -j\$(nproc)"
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
echo " Episodes   : ${EPISODES}"
echo " Sim time   : ${SIM_TIME}s"
echo " Sim log    : ${SIM_LOG}"
echo "============================================================"

# ── STEP 1: Start Python FIRST (it is the ZMQ server that NS-3 connects to) ──
#
# ns3-gym architecture:
#   Python Ns3Env     → ZMQ ROUTER (binds / listens on GYM_PORT)
#   C++ OpenGymInterface → ZMQ DEALER (connects to Python)
#
# Python must bind before NS-3 launches. If NS-3 starts first its connect()
# call times out immediately and the handshake never completes.
#
echo "[INFO] Starting Python training (background): ${PYTHON_SCRIPT} $*"
python "${PYTHON_SCRIPT}" \
    --port  "${GYM_PORT}" \
    --seed  "${SIM_SEED}" \
    "$@" &
PYTHON_PID=$!

# ── STEP 2: Wait until Python has bound the ZMQ port ─────────────────────────
echo -n "[INFO] Waiting for Python to open ZMQ port ${GYM_PORT}..."
WAIT_LIMIT=30
ELAPSED=0
while ! ss -tlnp 2>/dev/null | grep -q ":${GYM_PORT}"; do
    sleep 0.5
    ELAPSED=$(( ELAPSED + 1 ))

    # If Python died during startup (import error, missing dep, etc.) abort now.
    if ! kill -0 "${PYTHON_PID}" 2>/dev/null; then
        echo ""
        echo "[ERROR] Python process died before opening port ${GYM_PORT}."
        echo "        Check for import errors or config problems above."
        exit 1
    fi

    if (( ELAPSED >= WAIT_LIMIT * 2 )); then
        echo ""
        echo "[ERROR] Python did not open port ${GYM_PORT} within ${WAIT_LIMIT}s."
        kill "${PYTHON_PID}" 2>/dev/null || true
        exit 1
    fi
done
echo " ready."

# ── STEP 3: Launch NS-3 (connects to Python's ZMQ socket) ─────────────────────
echo "[INFO] Starting NS-3 simulation (background) — log: ${SIM_LOG}"
"${NS3_SCRATCH_BIN}" \
    --gymPort="${GYM_PORT}" \
    --seed="${SIM_SEED}" \
    --simTime="${SIM_TIME}" \
    > "${SIM_LOG}" 2>&1 &
NS3_PID=$!
echo "[INFO] NS-3 started — PID ${NS3_PID}"
echo "[INFO] Training in progress — waiting for Python (PID ${PYTHON_PID})..." ""this line was after stpe7 moved here for asstatic purposes
echo "============================================================"

# ── STEP 4: Verify NS-3 actually started (doesn't crash in first 3 s) ─────────
sleep 3
if ! kill -0 "${NS3_PID}" 2>/dev/null; then
    echo "[ERROR] NS-3 process exited within 3 s of launch."
    echo "        Last 20 lines of ${SIM_LOG}:"
    echo "------------------------------------------------------------"
    tail -n 20 "${SIM_LOG}" 2>/dev/null || echo "(log empty)"
    echo "------------------------------------------------------------"
    kill "${PYTHON_PID}" 2>/dev/null || true
    exit 1
fi
echo "[INFO] NS-3 alive after 3 s — handshake assumed successful."

# ── STEP 5: Tail NS-3 log in background so you can see C++ output live ────────
# Lines are prefixed with [NS3] so they're distinguishable from Python output.
# Only errors and warnings are forwarded to avoid flooding the terminal.
tail -F "${SIM_LOG}" 2>/dev/null \
    | grep --line-buffered -E "(ERROR|WARN|assert|Abort|terminate|ZMQ|GYM|Slice)" \
    | sed 's/^/[NS3] /' &
TAIL_PID=$!

# ── STEP 6: Periodic heartbeat — confirms both processes are alive ─────────────
# Runs in background; prints a status line every 5 minutes.
# Exits automatically once Python finishes.
(
    BEAT_INTERVAL=300   # seconds
    while kill -0 "${PYTHON_PID}" 2>/dev/null; do
        sleep "${BEAT_INTERVAL}"
        if ! kill -0 "${PYTHON_PID}" 2>/dev/null; then break; fi
        NS3_ALIVE="ON-LINE"
        if ! kill -0 "${NS3_PID}" 2>/dev/null; then NS3_ALIVE="OFF-LINE"; fi
        ELAPSED_MIN=$(( (SECONDS) / 60 ))
        echo "[HEARTBEAT] ${ELAPSED_MIN} min elapsed | Python ON-LINE | NS-3 ${NS3_ALIVE}"
        # If NS-3 dies mid-training, Python will hang on next env.step().
        # Log loudly so you notice.
        if [[ "${NS3_ALIVE}" == "OFF-LINE" ]]; then
            echo "[HEARTBEAT][ERROR] NS-3 OFF-LINE. Python will hang. Kill the script (Ctrl-C)."
        fi
    done
) &

# ── STEP 7: Wait for Python training to complete ──────────────────────────────

wait "${PYTHON_PID}"
PYTHON_EXIT=$?

if (( PYTHON_EXIT != 0 )); then
    echo "[WARN] Python exited with code ${PYTHON_EXIT}."
else
    echo "[INFO] Python training completed successfully."
fi

echo "============================================================"
echo "[INFO] NS-3 log: ${SIM_LOG}"
echo "[INFO] Models:   results/models/"
echo "[INFO] Logs:     results/logs/"
echo "[INFO] Run tensorboard with:"
echo "       tensorboard --logdir results/tensorboard"
echo "============================================================"

# cleanup() fires via the EXIT trap — kills NS-3 and the tail process.
