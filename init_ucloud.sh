#!/bin/bash
# ============================================================
# UCloud Initialization Script
# Point UCloud's "Initialization" field at this file.
# It runs automatically when the job starts.
#
# Optional: set env vars via UCloud's "Extra options" field, e.g.:
#   HF_TOKEN=hf_xxx CPU_TEST=0 MODEL_NAME=meta-llama/Llama-2-7b-hf
# ============================================================
set -euo pipefail

REPO_DIR="/work/LUAS"
LOG_FILE="${REPO_DIR}/training_run.log"

echo "========== UCloud init started at $(date) ==========" | tee -a "${LOG_FILE}"

# ── 1. Pull latest code ──────────────────────────────────────
cd "${REPO_DIR}"
git config --global pull.rebase true
git config --global rebase.autoStash true
git pull --rebase origin main 2>&1 | tee -a "${LOG_FILE}"

# ── 2. Activate virtualenv ───────────────────────────────────
source "${REPO_DIR}/.venv/bin/activate"

# ── 3. Export training env vars (override via "Extra options") ─
export HF_TOKEN="${HF_TOKEN:-}"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
export LOG_EVERY="${LOG_EVERY:-10}"
export DATASET_DIR="${DATASET_DIR:-../generation/multiwoz/converters/woz.2.2.gen}"
export CPU_TEST="${CPU_TEST:-1}"           # 1 = smoke-test (fast), 0 = full run
export MODEL_NAME="${MODEL_NAME:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"

echo "MODEL_NAME: ${MODEL_NAME}"          | tee -a "${LOG_FILE}"
echo "CPU_TEST:   ${CPU_TEST}"            | tee -a "${LOG_FILE}"
echo "DATASET_DIR:${DATASET_DIR}"         | tee -a "${LOG_FILE}"

# ── 4. Run training (output goes to log + terminal) ──────────
echo "========== Starting training at $(date) ==========" | tee -a "${LOG_FILE}"
bash "${REPO_DIR}/train_ucloud.sh" 2>&1 | tee -a "${LOG_FILE}"

echo "========== Training finished at $(date) ==========" | tee -a "${LOG_FILE}"
