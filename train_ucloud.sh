#!/bin/bash
set -x

# Configuration - modify based on your UCloud setup
NUM_GPUS=4  # Change this to match your GPU count (1, 2, or 4)
DATASET_NAME="agent_sft_act_dataset"

export PYTHONPATH=`pwd`
export HF_TOKEN=${HF_TOKEN:-"your_huggingface_token_here"}
export HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN:-"your_huggingface_token_here"}

echo "PYTHONPATH: ${PYTHONPATH}"
echo "NUM_GPUS: ${NUM_GPUS}"

cd ./training_scripts

MODEL_TYPE="7b"
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
        --model_name meta-llama/Llama-2-${MODEL_TYPE}-hf \
        --use_peft \
        --peft_method lora \
        --output_dir ${SAVE_DIR} \
        --pure_bf16 \
        --dataset ${DATASET_NAME} \
        --dataset_type gen \
        --batch_size_training ${BATCH_SIZE} \
        --num_epochs ${EPOCH} \
        --lr ${LR}
else
    # Single GPU training without FSDP
    python llama_finetuning.py \
        --model_name meta-llama/Llama-2-${MODEL_TYPE}-hf \
        --use_peft \
        --peft_method lora \
        --output_dir ${SAVE_DIR} \
        --pure_bf16 \
        --dataset ${DATASET_NAME} \
        --dataset_type gen \
        --batch_size_training ${BATCH_SIZE} \
        --num_epochs ${EPOCH} \
        --lr ${LR}
fi

# Convert FSDP checkpoint to HuggingFace format (only needed for multi-GPU)
if [ "$USE_FSDP" = true ]; then
    echo "========== Converting Stage 1 checkpoint =========="
    python inference/checkpoint_converter_fsdp_hf.py \
        --fsdp_checkpoint_path ${SAVE_DIR} \
        --consolidated_model_path ${SAVE_DIR}-HF \
        --HF_model_path_or_name meta-llama/Llama-2-${MODEL_TYPE}-hf
    
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
        --dataset_type real \
        --batch_size_training ${BATCH_SIZE} \
        --num_epochs ${EPOCH} \
        --lr ${LR}
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
