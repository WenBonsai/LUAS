# UCloud Training Setup Guide for LUAS Project

## Prerequisites
- UCloud account with GPU allocation
- Project uploaded to UCloud workspace

## Quick Start

### 1. Environment Setup

```bash
cd LUAS
python3 -m venv .venv
source .venv/bin/activate

# Install PyTorch with CUDA support
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install transformers==4.34.0 accelerate==0.23.0 peft datasets
pip install bitsandbytes sentencepiece protobuf tiktoken openai wandb

# Set HuggingFace token (replace with YOUR token)
export HF_TOKEN=your_huggingface_token_here
export HUGGING_FACE_HUB_TOKEN=your_huggingface_token_here
```

### 2. Verify Data

```bash
# Check synthetic data (should be ~58MB)
ls -lh generation/multiwoz/converters/woz.2.2.gen/train.act.json

# Check real data (should be ~9.4MB)
ls -lh generation/multiwoz/converters/woz.2.2.real/train.act.json
```

### 3. Configure Training Script

Edit `train_ucloud.sh` and set `NUM_GPUS` to match your allocation:

```bash
NUM_GPUS=4  # Change to 1, 2, or 4 based on your UCloud setup
```

### 4. Run Training

```bash
# Make script executable
chmod +x train_ucloud.sh

# Start training
bash train_ucloud.sh
```

## Training Process

The script performs **two-stage training**:

### Stage 1: Synthetic Data (51K dialogues)
- Trains LLaMA-2-7B on generated synthetic dialogues
- Creates: `training_scripts/agent_sft_act_dataset.7b.2e-5.full.B4.E1.gen/`
- Duration: ~8-12 hours on 4 GPUs

### Stage 2: Real Data (8K dialogues)  
- Fine-tunes Stage 1 model on real MultiWOZ 2.2 data
- Creates: `training_scripts/agent_sft_act_dataset.7b.2e-5.full.B4.E1.real/`
- Duration: ~2-3 hours on 4 GPUs

## Monitoring Training

### View logs in real-time:
```bash
tail -f training_scripts/train_output.log
```

### Check GPU usage:
```bash
watch -n 1 nvidia-smi
```

### Monitor with Weights & Biases (optional):
Set `WANDB_PROJECT` in `train_ucloud.sh` and run:
```bash
wandb login
```

## Troubleshooting

### Out of Memory (OOM) errors:
Reduce `BATCH_SIZE` in `train_ucloud.sh`:
```bash
BATCH_SIZE=2  # or 1 if still OOM
```

### Connection timeout:
If model download fails, manually download first:
```bash
python -c "from transformers import LlamaForCausalLM, LlamaTokenizer; \
    LlamaForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf', token='YOUR_TOKEN'); \
    LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', token='YOUR_TOKEN')"
```

### GPU not detected:
```bash
nvidia-smi  # Check if GPUs are visible
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

## Expected Timeline

- **4 GPUs**: ~10-15 hours total (Stage 1: 8-12h, Stage 2: 2-3h)
- **2 GPUs**: ~20-30 hours total
- **1 GPU**: ~40-60 hours total

## After Training

### Convert FSDP checkpoint to HuggingFace format (if using multi-GPU):
The script does this automatically between stages.

### Run Evaluation:
```bash
cd eval
bash start_svr_vllm.sh  # Start inference server
python run_inference_vllm.py  # Generate predictions
python metric.py  # Compute JGA metric
```

## Tips

- Use `screen` or `tmux` to keep training running if SSH disconnects
- Monitor `nvidia-smi` to ensure GPUs are being utilized
- Check disk space regularly - training generates ~20GB of checkpoints
- Save your token in `.bashrc` or `.zshrc` for convenience (but don't commit it!)

## Support

For issues, consult:
- Original paper: [ArXiv link in README.md]
- MultiWOZ dataset: https://github.com/budzianowski/multiwoz
- HuggingFace documentation: https://huggingface.co/docs
