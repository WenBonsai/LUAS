#!/bin/bash
# Interactive training script with environment checks
# Usage: bash train_interactive.sh [num_gpus] [batch_size]

set -e

# Configuration
NUM_GPUS=${1:-1}
BATCH_SIZE=${2:-4}

echo "=========================================="
echo "LUAS Interactive Training"
echo "=========================================="
echo "Configuration:"
echo "  GPUs: $NUM_GPUS"
echo "  Batch Size: $BATCH_SIZE"
echo "=========================================="

# Environment checks
echo "Checking environment..."

# Check virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠ Virtual environment not activated"
    if [ -d ".venv" ]; then
        echo "Activating .venv..."
        source .venv/bin/activate
    else
        echo "ERROR: No virtual environment found. Run: bash setup_env.sh"
        exit 1
    fi
else
    echo "✓ Virtual environment: $VIRTUAL_ENV"
fi

# Check HuggingFace token
if [ -z "$HF_TOKEN" ]; then
    echo ""
    echo "⚠ HF_TOKEN not set!"
    read -p "Enter your HuggingFace token (or press Enter to skip): " input_token
    if [ -n "$input_token" ]; then
        export HF_TOKEN="$input_token"
        export HUGGING_FACE_HUB_TOKEN="$input_token"
        echo "✓ Token set"
    else
        echo "WARNING: Continuing without token - this may fail!"
    fi
else
    echo "✓ HuggingFace token set"
fi

# Check CUDA
python -c "import torch; cuda_available = torch.cuda.is_available(); gpu_count = torch.cuda.device_count() if cuda_available else 0; print(f'✓ CUDA: {cuda_available}, GPUs: {gpu_count}'); exit(0 if cuda_available or True else 1)"

# Check data
echo ""
echo "Checking datasets..."
if [ ! -f "generation/multiwoz/converters/woz.2.2.gen/train.act.json" ]; then
    echo "ERROR: Synthetic data not found at generation/multiwoz/converters/woz.2.2.gen/train.act.json"
    exit 1
fi
echo "✓ Synthetic data found"

if [ ! -f "generation/multiwoz/converters/woz.2.2.real/train.act.json" ]; then
    echo "ERROR: Real data not found at generation/multiwoz/converters/woz.2.2.real/train.act.json"
    exit 1
fi
echo "✓ Real data found"

# Update training script with parameters
echo ""
echo "Configuring training script..."
sed -i.bak "s/NUM_GPUS=.*/NUM_GPUS=$NUM_GPUS/" train_ucloud.sh
sed -i.bak "s/BATCH_SIZE=.*/BATCH_SIZE=$BATCH_SIZE/" train_ucloud.sh
echo "✓ Configuration updated"

# Ask for confirmation
echo ""
echo "=========================================="
echo "Ready to start training!"
echo "=========================================="
echo "This will run a two-stage training:"
echo "  Stage 1: Synthetic data (~8-12 hours on 4 GPUs)"
echo "  Stage 2: Real data (~2-3 hours on 4 GPUs)"
echo ""
read -p "Continue? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training cancelled"
    exit 0
fi

# Create logs directory
mkdir -p logs
mkdir -p training_scripts

# Run training
echo ""
echo "=========================================="
echo "Starting training..."
echo "=========================================="
LOG_FILE="logs/training_$(date +%Y%m%d_%H%M%S).log"
bash train_ucloud.sh 2>&1 | tee "$LOG_FILE"

echo ""
echo "=========================================="
echo "Training completed!"
echo "Log saved to: $LOG_FILE"
echo "=========================================="
