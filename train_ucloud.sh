#!/bin/bash
set -euo pipefail

if [[ "${DEBUG:-0}" == "1" ]]; then
    set -x
fi

# Configuration - modify based on your UCloud setup
NUM_GPUS=1  # Change this to match your GPU count (1, 2, or 4)
DATASET_NAME="agent_sft_act_dataset"

# Model to train. For GPU jobs use: meta-llama/Llama-2-7b-hf
# For CPU smoke-tests use TinyLlama (public, no token needed, ~600MB)
MODEL_NAME=${MODEL_NAME:-"TinyLlama/TinyLlama-1.1B-Chat-v1.0"}

# CPU thread count — set to number of vCPUs on the machine
NUM_THREADS=${NUM_THREADS:-$(nproc)}

# Directory that contains train.act.json (relative to ./training_scripts after cd)
# Note: repo root also contains `woz.2.2.gen/train.act.json` but it may be empty.
DATASET_DIR=${DATASET_DIR:-"../generation/multiwoz/converters/woz.2.2.gen"}

# How often to checkpoint and evaluate (in training steps)
CKPT_STEPS=${CKPT_STEPS:-100}
EVAL_STEPS=${EVAL_STEPS:-200}

# CPU_TEST=1 → tiny smoke-test (50 train samples, no eval, fast finish)
# Set CPU_TEST=0 (default) for a real training run
CPU_TEST=${CPU_TEST:-1}
if [[ "${CPU_TEST}" == "1" ]]; then
    echo "[CPU_TEST mode] Using tiny dataset for fast smoke-test. Set CPU_TEST=0 for full training."
    MAX_TRAIN=${MAX_TRAIN:-50}
    MAX_VAL=${MAX_VAL:-0}       # 0 = skip validation entirely
    CKPT_STEPS=999999            # effectively disable mid-run checkpoints
    EVAL_STEPS=999999            # effectively disable mid-run eval
    RUN_VALIDATION="--run_validation False"
else
    MAX_TRAIN=${MAX_TRAIN:--1}   # -1 = full dataset
    MAX_VAL=${MAX_VAL:--1}
    RUN_VALIDATION=""
fi

export PYTHONPATH=`pwd`
export OMP_NUM_THREADS=${NUM_THREADS}
export MKL_NUM_THREADS=${NUM_THREADS}
export TORCH_NUM_THREADS=${NUM_THREADS}

if [[ -z "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
    echo "Warning: HF_TOKEN/HUGGING_FACE_HUB_TOKEN not set; gated models (e.g., Llama-2) will fail to download." >&2
fi

echo "PYTHONPATH: ${PYTHONPATH}"
echo "NUM_GPUS: ${NUM_GPUS}"
echo "MODEL: ${MODEL_NAME}"
echo "CPU threads: ${NUM_THREADS}"

cd ./training_scripts

if [[ ! -s "${DATASET_DIR}/train.act.json" ]]; then
    echo "Error: dataset file missing or empty: ${DATASET_DIR}/train.act.json" >&2
    echo "Tip: set DATASET_DIR to the folder containing train.act.json, e.g.:" >&2
    echo "  DATASET_DIR=../generation/multiwoz/converters/woz.2.2.gen bash train_ucloud.sh" >&2
    exit 1
fi

MODEL_TYPE="1b"  # used only in output dir naming
LR=2e-5
BATCH_SIZE=4
EPOCH=1

# Adjust CUDA_VISIBLE_DEVICES based on NUM_GPUS
case $NUM_GPUS in
    1)
        export CUDA_VISIBLE_DEVICES=0
        USE_FSDP=false
        ;;
    2)
        export CUDA_VISIBLE_DEVICES=0,1
        USE_FSDP=true
        ;;
    4)
        export CUDA_VISIBLE_DEVICES=0,1,2,3
        USE_FSDP=true
        ;;
    *)
        echo "Error: NUM_GPUS must be 1, 2, or 4"
        exit 1
        ;;
esac

echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "USE_FSDP: ${USE_FSDP}"

# Stage 1: Train on synthetic data
echo "========== Stage 1: Training on synthetic data =========="

SAVE_DIR="${DATASET_NAME}.${MODEL_TYPE}.${LR}.full.B${BATCH_SIZE}.E${EPOCH}.gen"

if [ "$USE_FSDP" = true ]; then
    # Multi-GPU training with FSDP
    torchrun \
        --nnodes 1 \
        --nproc_per_node=$NUM_GPUS \
        llama_finetuning.py \
        --enable_fsdp \
        --model_name ${MODEL_NAME} \
        --use_peft \
        --peft_method lora \
        --output_dir ${SAVE_DIR} \
        --pure_bf16 \
        --dataset ${DATASET_NAME} \
        --dataset_dir ${DATASET_DIR} \
        --dataset_type gen \
        --batch_size_training ${BATCH_SIZE} \
        --num_epochs ${EPOCH} \
        --lr ${LR}
else
    # Single GPU training without FSDP
    python llama_finetuning.py \
        --model_name ${MODEL_NAME} \
        --use_peft \
        --peft_method lora \
        --output_dir ${SAVE_DIR} \
        --pure_bf16 \
        --dataset ${DATASET_NAME} \
        --dataset_dir ${DATASET_DIR} \
        --dataset_type gen \
        --batch_size_training ${BATCH_SIZE} \
        --num_epochs ${EPOCH} \
        --lr ${LR} \
        --check_point_steps ${CKPT_STEPS} \
        --evaluation_steps ${EVAL_STEPS} \
        --max_train_samples ${MAX_TRAIN} \
        --max_val_samples ${MAX_VAL} \
        ${RUN_VALIDATION}
fi

# Convert FSDP checkpoint to HuggingFace format (only needed for multi-GPU)
if [ "$USE_FSDP" = true ]; then
    echo "========== Converting Stage 1 checkpoint =========="
    python inference/checkpoint_converter_fsdp_hf.py \
        --fsdp_checkpoint_path ${SAVE_DIR} \
        --consolidated_model_path ${SAVE_DIR}-HF \
        --HF_model_path_or_name ${MODEL_NAME}
    
    STAGE1_MODEL="${SAVE_DIR}-HF"
else
    STAGE1_MODEL="${SAVE_DIR}"
fi

# Stage 2: Fine-tune on real data
echo "========== Stage 2: Fine-tuning on real data =========="

SAVE_DIR_REAL="${DATASET_NAME}.${MODEL_TYPE}.${LR}.full.B${BATCH_SIZE}.E${EPOCH}.real"

if [ "$USE_FSDP" = true ]; then
    # Multi-GPU training with FSDP
    torchrun \
        --nnodes 1 \
        --nproc_per_node=$NUM_GPUS \
        llama_finetuning.py \
        --enable_fsdp \
        --model_name ${STAGE1_MODEL} \
        --use_peft \
        --peft_method lora \
        --output_dir ${SAVE_DIR_REAL} \
        --pure_bf16 \
        --dataset ${DATASET_NAME} \
        --dataset_dir ${DATASET_DIR} \
        --dataset_type real \
        --batch_size_training ${BATCH_SIZE} \
        --num_epochs ${EPOCH} \
        --lr ${LR}
else
    # Single GPU training without FSDP
    python llama_finetuning.py \
        --model_name ${STAGE1_MODEL} \
        --use_peft \
        --peft_method lora \
        --output_dir ${SAVE_DIR_REAL} \
        --pure_bf16 \
        --dataset ${DATASET_NAME} \
        --dataset_dir ${DATASET_DIR} \
        --dataset_type real \
        --batch_size_training ${BATCH_SIZE} \
        --num_epochs ${EPOCH} \
        --lr ${LR} \
        --check_point_steps ${CKPT_STEPS} \
        --evaluation_steps ${EVAL_STEPS} \
        --max_train_samples ${MAX_TRAIN} \
        --max_val_samples ${MAX_VAL} \
        ${RUN_VALIDATION}
fi

# Convert final checkpoint (only needed for multi-GPU)
if [ "$USE_FSDP" = true ]; then
    echo "========== Converting Stage 2 checkpoint =========="
    python inference/checkpoint_converter_fsdp_hf.py \
        --fsdp_checkpoint_path ${SAVE_DIR_REAL} \
        --consolidated_model_path ${SAVE_DIR_REAL}-HF \
        --HF_model_path_or_name ${STAGE1_MODEL}
    
    echo "Training complete! Final model saved to: ${SAVE_DIR_REAL}-HF"
else
    echo "Training complete! Final model saved to: ${SAVE_DIR_REAL}"
fi
