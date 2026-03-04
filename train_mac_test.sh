#!/bin/bash
set -x

# Small test training for Mac (no FSDP, single device)
DATASET_NAME="agent_sft_act_dataset"

export PYTHONPATH=`pwd`
export HF_TOKEN=${HF_TOKEN:-"your_huggingface_token_here"}
export HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN:-"your_huggingface_token_here"}
echo ${PYTHONPATH}

cd ./training_scripts

# Use a smaller model or adjust if you have LLaMA-2-7B
MODEL_TYPE="7b"
MODEL_NAME="meta-llama/Llama-2-${MODEL_TYPE}-hf"

# Small test dataset
DATASET_DIR="../generation/multiwoz/converters/woz.2.2.gen.small/"

LR=2e-5
BATCH_SIZE=1  # Very small for Mac
EPOCH=1

TAG="mac_test.${MODEL_TYPE}.${LR}.B${BATCH_SIZE}.E${EPOCH}"

# Use venv Python
PYTHON_CMD="/Users/wenlong/Documents/GitHub/LUAS/.venv/bin/python"

# Single device training (no FSDP)
${PYTHON_CMD} ./llama_finetuning.py \
  --model_name "${MODEL_NAME}" \
  --dataset "${DATASET_NAME}" \
  --dataset_dir "${DATASET_DIR}" \
  --save_model \
  --output_dir "./${DATASET_NAME}.${TAG}"/ \
  --lr ${LR} \
  --valid_batch_size ${BATCH_SIZE} \
  --train_batch_size ${BATCH_SIZE} \
  --micro_batch_size ${BATCH_SIZE} \
  --num_epochs ${EPOCH} \
  --evaluation_steps 50 \
  --check_point_steps 1000000

echo "Stage 1 (synthetic data) complete!"
echo "For Stage 2, you would train on woz.2.2.real.small/"
