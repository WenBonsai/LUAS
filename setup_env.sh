#!/bin/bash
# Quick environment setup script for LUAS training
# Usage: bash setup_env.sh

set -e  # Exit on error

echo "=========================================="
echo "LUAS Environment Setup"
echo "=========================================="

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $PYTHON_VERSION"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
source .venv/bin/activate
echo "✓ Virtual environment activated"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip -q

# Detect CUDA availability
if command -v nvidia-smi &> /dev/null; then
    echo "✓ CUDA detected - installing PyTorch with CUDA support"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "⚠ No CUDA detected - installing CPU-only PyTorch"
    pip install torch torchvision torchaudio
fi

# Install dependencies
echo "Installing Python dependencies..."
pip install transformers==4.34.0 accelerate==0.23.0 peft datasets bitsandbytes -q
pip install fire sentencepiece protobuf tiktoken fuzzywuzzy tqdm openai -q

echo ""
echo "=========================================="
echo "✓ Environment setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Set your HuggingFace token:"
echo "   export HF_TOKEN=your_token_here"
echo ""
echo "2. Verify GPU access (if on cloud):"
echo "   python -c 'import torch; print(f\"CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}\")'"
echo ""
echo "3. Run training:"
echo "   bash train_ucloud.sh"
echo ""
