#!/bin/bash
# ============================================================
# UCloud Initialization Script — fresh Ubuntu environment
# Point UCloud's "Initialization" field at this file.
#
# This script runs on a brand-new machine with nothing installed.
# It will:
#   1. Install system deps + Python
#   2. Clone the LUAS repo from GitHub
#   3. Create a venv and install Python packages
#   4. Run training automatically
#
# Override any variable via UCloud's "Extra options" field, e.g.:
#   HF_TOKEN=hf_xxx CPU_TEST=0 MODEL_NAME=meta-llama/Llama-2-7b-hf
# ============================================================
set -euo pipefail

REPO_URL="https://github.com/WenBonsai/LUAS.git"
REPO_DIR="/home/ucloud/LUAS"
LOG_FILE="/home/ucloud/training_run.log"
PYTHON_BIN="python3"

echo "========== UCloud init started at $(date) ==========" | tee -a "${LOG_FILE}"

# ── 1. System dependencies ───────────────────────────────────
echo "[1/5] Installing system packages..." | tee -a "${LOG_FILE}"
apt-get update -qq
apt-get install -y -qq git python3 python3-pip python3-venv curl 2>&1 | tee -a "${LOG_FILE}"

# ── 2. Clone repo (or pull if already exists) ────────────────
echo "[2/5] Cloning repo..." | tee -a "${LOG_FILE}"
if [[ -d "${REPO_DIR}/.git" ]]; then
    cd "${REPO_DIR}"
    git config --global pull.rebase true
    git pull --rebase origin main 2>&1 | tee -a "${LOG_FILE}"
else
    git clone "${REPO_URL}" "${REPO_DIR}" 2>&1 | tee -a "${LOG_FILE}"
    cd "${REPO_DIR}"
fi

# ── 3. Create virtualenv ─────────────────────────────────────
echo "[3/5] Setting up Python virtualenv..." | tee -a "${LOG_FILE}"
if [[ ! -f "${REPO_DIR}/.venv/bin/activate" ]]; then
    ${PYTHON_BIN} -m venv "${REPO_DIR}/.venv" 2>&1 | tee -a "${LOG_FILE}"
fi
source "${REPO_DIR}/.venv/bin/activate"

# ── 4. Install Python dependencies ───────────────────────────
echo "[4/5] Installing Python packages (this takes a few minutes)..." | tee -a "${LOG_FILE}"
pip install --upgrade pip --quiet
pip install -r "${REPO_DIR}/requirements.txt" --quiet 2>&1 | tee -a "${LOG_FILE}"

# ── 5. Run training ──────────────────────────────────────────
echo "[5/5] Starting training at $(date)..." | tee -a "${LOG_FILE}"

export HF_TOKEN="${HF_TOKEN:-}"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
export LOG_EVERY="${LOG_EVERY:-10}"
export DATASET_DIR="${DATASET_DIR:-../generation/multiwoz/converters/woz.2.2.gen}"
export CPU_TEST="${CPU_TEST:-1}"           # 1 = TinyLlama smoke-test, 0 = full Llama-2 run
export MODEL_NAME="${MODEL_NAME:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"

echo "MODEL:      ${MODEL_NAME}"          | tee -a "${LOG_FILE}"
echo "CPU_TEST:   ${CPU_TEST}"            | tee -a "${LOG_FILE}"
echo "DATASET_DIR:${DATASET_DIR}"         | tee -a "${LOG_FILE}"

bash "${REPO_DIR}/train_ucloud.sh" 2>&1 | tee -a "${LOG_FILE}"

echo "========== Training finished at $(date) ==========" | tee -a "${LOG_FILE}"
