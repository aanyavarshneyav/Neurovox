#!/usr/bin/env bash
# run_headline.sh — Sequential S0 and S1 headline training on Apple Silicon (MPS)
# Usage: ./scripts/run_headline.sh [seed]
# Default seed is 42 if omitted.

set -euo pipefail

SEED="${1:-42}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

# Resolve Python environment
if [ -f ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON_BIN="python3"
else
    PYTHON_BIN="python"
fi

# Ensure MPS operator fallback is enabled for unsupported ops (Rule 5.4)
export PYTORCH_ENABLE_MPS_FALLBACK=1

echo "================================================================================"
echo "NEUROVOX HEADLINE SEQUENTIAL TRAINING"
echo "Seed:            ${SEED}"
echo "Platform:        Mac M3 Pro (device=mps, fp32)"
echo "Python:          $(${PYTHON_BIN} --version) (${PYTHON_BIN})"
echo "MPS Fallback:    PYTORCH_ENABLE_MPS_FALLBACK=${PYTORCH_ENABLE_MPS_FALLBACK}"
echo "================================================================================"

# -----------------------------------------------------------------------------
# Stage 1: S0 Headline Run (IAAMSoftmax with AdaptiveDrop)
# -----------------------------------------------------------------------------
S0_RUN_ID="s0_seed${SEED}"
S0_CKPT_DIR="checkpoints/${S0_RUN_ID}"
S0_RESUME=""

if [ -d "${S0_CKPT_DIR}" ] && ls "${S0_CKPT_DIR}"/epoch_*.pt 1>/dev/null 2>&1; then
    echo "[run_headline] Checkpoints found in ${S0_CKPT_DIR}; resuming training..."
    S0_RESUME="--resume"
fi

echo ""
echo ">>> Starting S0 headline run: ${S0_RUN_ID}"
caffeinate -ims "${PYTHON_BIN}" scripts/train.py \
    --config configs/s0_full.yaml \
    --run-id "${S0_RUN_ID}" \
    --seed "${SEED}" \
    --device mps \
    ${S0_RESUME}

echo ">>> S0 headline run (${S0_RUN_ID}) completed."

# -----------------------------------------------------------------------------
# Stage 2: S1 Headline Run (Standard AAMSoftmax baseline)
# -----------------------------------------------------------------------------
S1_RUN_ID="s1_seed${SEED}"
S1_CKPT_DIR="checkpoints/${S1_RUN_ID}"
S1_RESUME=""

if [ -d "${S1_CKPT_DIR}" ] && ls "${S1_CKPT_DIR}"/epoch_*.pt 1>/dev/null 2>&1; then
    echo "[run_headline] Checkpoints found in ${S1_CKPT_DIR}; resuming training..."
    S1_RESUME="--resume"
fi

echo ""
echo ">>> Starting S1 headline run: ${S1_RUN_ID}"
caffeinate -ims "${PYTHON_BIN}" scripts/train.py \
    --config configs/s1_full.yaml \
    --run-id "${S1_RUN_ID}" \
    --seed "${SEED}" \
    --device mps \
    ${S1_RESUME}

echo ">>> S1 headline run (${S1_RUN_ID}) completed."
echo "================================================================================"
echo "ALL HEADLINE RUNS FOR SEED ${SEED} COMPLETED SUCCESSFULLY."
echo "================================================================================"
