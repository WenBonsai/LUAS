#!/bin/sh
# ============================================================
# UCloud Initialization Script — Qwen2 LoRA DST Training
# ============================================================
# Usage: Upload this file to UCloud and point the job's
#        "Initialization script" field at it, OR run manually:
#
#   bash init_ucloud.sh
#
# Optional environment overrides:
#   MODEL_NAME=Qwen/Qwen2-1.5B-Instruct bash init_ucloud.sh
#   MAX_STEPS=2400 bash init_ucloud.sh
#   HF_TOKEN=hf_xxx bash init_ucloud.sh
# ============================================================
set -eu

# ── Configuration (override via env) ─────────────────────────
REPO_URL="https://github.com/WenBonsai/LUAS.git"
REPO_DIR="${HOME}/LUAS"
LOG_FILE="${HOME}/init_ucloud.log"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2-0.5B-Instruct}"
MAX_STEPS="${MAX_STEPS:-10}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-256}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/qwen_lora_ucloud}"
HF_TOKEN="${HF_TOKEN:-}"

echo "========================================" | tee "${LOG_FILE}"
echo " UCloud init started at $(date)"         | tee -a "${LOG_FILE}"
echo " MODEL:     ${MODEL_NAME}"               | tee -a "${LOG_FILE}"
echo " MAX_STEPS: ${MAX_STEPS}"               | tee -a "${LOG_FILE}"
echo "========================================"| tee -a "${LOG_FILE}"

# ── 1. System dependencies ───────────────────────────────────
echo "[1/5] Installing system packages..." | tee -a "${LOG_FILE}"
sudo apt-get update -qq
sudo apt-get install -y -qq git python3 python3-pip python3-venv curl

# ── 2. Clone or update repo ──────────────────────────────────
echo "[2/5] Cloning repo..." | tee -a "${LOG_FILE}"
if [ -d "${REPO_DIR}/.git" ]; then
    cd "${REPO_DIR}"
    git pull --rebase origin main 2>&1 | tee -a "${LOG_FILE}"
else
    git clone "${REPO_URL}" "${REPO_DIR}" 2>&1 | tee -a "${LOG_FILE}"
    cd "${REPO_DIR}"
fi

# ── 3. Create virtualenv ─────────────────────────────────────
echo "[3/5] Setting up Python venv..." | tee -a "${LOG_FILE}"
if [ ! -f "${REPO_DIR}/.venv/bin/activate" ]; then
    python3 -m venv "${REPO_DIR}/.venv"
fi
source "${REPO_DIR}/.venv/bin/activate"

# ── 4. Install Python dependencies ───────────────────────────
echo "[4/5] Installing Python packages..." | tee -a "${LOG_FILE}"
pip install --upgrade pip --quiet

# Auto-detect CUDA and install matching torch
if nvidia-smi >/dev/null 2>&1; then
    CUDA_VER=$(nvidia-smi | grep -oP "CUDA Version: \K[\d]+" | head -1)
    echo "  GPU detected — CUDA ${CUDA_VER}" | tee -a "${LOG_FILE}"
    if [ "${CUDA_VER}" -ge 12 ]; then
        pip install torch --index-url https://download.pytorch.org/whl/cu121 --quiet
    else
        pip install torch --index-url https://download.pytorch.org/whl/cu118 --quiet
    fi
else
    echo "  No GPU — installing CPU torch" | tee -a "${LOG_FILE}"
    pip install torch --index-url https://download.pytorch.org/whl/cpu --quiet
fi

pip install transformers peft datasets accelerate sentencepiece protobuf --quiet
echo "  Packages installed OK" | tee -a "${LOG_FILE}"

# ── 5. Run training ──────────────────────────────────────────
echo "[5/5] Starting training at $(date)..." | tee -a "${LOG_FILE}"

# Set HF token if provided
if [ -n "${HF_TOKEN}" ]; then
    export HF_TOKEN="${HF_TOKEN}"
    export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
fi

mkdir -p "${REPO_DIR}/${OUTPUT_DIR}"

python3 -u scripts/train_qwen_lora.py \
    --train_file data_full/train.jsonl \
    --dev_file   data_full/dev.jsonl \
    --model_name "${MODEL_NAME}" \
    --max_steps  "${MAX_STEPS}" \
    --max_seq_len "${MAX_SEQ_LEN}" \
    --output_dir  "${OUTPUT_DIR}" \
    2>&1 | tee -a "${LOG_FILE}"

echo "========================================" | tee -a "${LOG_FILE}"
echo " Training finished at $(date)"           | tee -a "${LOG_FILE}"
echo " Adapter saved to: ${OUTPUT_DIR}/lora_adapter" | tee -a "${LOG_FILE}"
echo "========================================"| tee -a "${LOG_FILE}"

# ── 6. Run evaluation ────────────────────────────────────────
echo "[+] Running evaluation..." | tee -a "${LOG_FILE}"

python3 -u scripts/eval_qwen_lora.py \
    --model_name  "${MODEL_NAME}" \
    --adapter_dir "${OUTPUT_DIR}/lora_adapter" \
    --data_file   data_full/dev.jsonl \
    --max_examples 0 \
    --max_new_tokens 128 \
    --dst_metrics \
    2>&1 | tee -a "${LOG_FILE}"

echo "========================================" | tee -a "${LOG_FILE}"
echo " All done at $(date)"                    | tee -a "${LOG_FILE}"
echo " Full log: ${LOG_FILE}"                  | tee -a "${LOG_FILE}"
echo "========================================"| tee -a "${LOG_FILE}"

# ── 7. Save log to GitHub ────────────────────────────────────
echo "[+] Saving log to GitHub..." | tee -a "${LOG_FILE}"
cd "${REPO_DIR}"
LOG_DEST="logs/$(date +%Y%m%d_%H%M%S)_ucloud.log"
mkdir -p logs
cp "${LOG_FILE}" "${LOG_DEST}"
git config user.email "ucloud@job"
git config user.name "UCloud Job"
git add "${LOG_DEST}"
git commit -m "Add training log: ${LOG_DEST}" 2>&1 | tee -a "${LOG_FILE}"
git push origin main 2>&1 | tee -a "${LOG_FILE}"
echo "[+] Log saved to repo: ${LOG_DEST}" | tee -a "${LOG_FILE}"
