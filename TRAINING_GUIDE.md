# LUAS Training Environment - Quick Reference

## Quick Start (Any Platform)

### 1. Setup Environment
```bash
bash setup_env.sh
source .venv/bin/activate
```

### 2. Set Token
```bash
export HF_TOKEN=your_huggingface_token_here
```

### 3. Run Training
```bash
# Interactive mode (recommended for first run)
bash train_interactive.sh

# Or specify GPUs and batch size
bash train_interactive.sh 4 4  # 4 GPUs, batch size 4
```

---

## Job Submission Methods

### Method 1: Interactive Training (Local/Cloud VM)
Best for: Direct control, debugging, small experiments
```bash
# Activate environment
source .venv/bin/activate
export HF_TOKEN=your_token

# Run with automatic checks
bash train_interactive.sh 4 4

# Or run directly
bash train_ucloud.sh
```

### Method 2: Background Job (UCloud/VM without job queue)
Best for: Long-running jobs without SLURM
```bash
# Submit background job
export HF_TOKEN=your_token
bash submit_ucloud_job.sh

# Monitor
tail -f logs/luas-training-*.log

# Check status
ps aux | grep train_ucloud
```

### Method 3: SLURM Job Queue (HPC Clusters)
Best for: University clusters, HPC systems
```bash
# Set token in environment or script
export HF_TOKEN=your_token

# Submit to queue
sbatch submit_job.sh

# Check status
squeue -u $USER

# View output
tail -f logs/luas_*.out
```

---

## Environment Variables

Required:
- `HF_TOKEN` - Your HuggingFace access token

Optional:
- `NUM_GPUS` - Number of GPUs to use (default: 4)
- `BATCH_SIZE` - Training batch size (default: 4)
- `WANDB_PROJECT` - For W&B logging
- `CUDA_VISIBLE_DEVICES` - Specific GPU selection

---

## Directory Structure

```
LUAS/
├── setup_env.sh              # Environment setup
├── train_interactive.sh      # Interactive training with checks
├── train_ucloud.sh          # Main training script
├── submit_job.sh            # SLURM job submission
├── submit_ucloud_job.sh     # Background job submission
├── requirements.minimal.txt  # Minimal dependencies
├── logs/                    # Training logs
└── training_scripts/        # Output checkpoints
```

---

## Common Commands

### Check Environment
```bash
# Verify Python packages
pip list | grep -E "(torch|transformers|accelerate|peft)"

# Check CUDA
nvidia-smi
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

# Test HF token
python -c "from huggingface_hub import model_info; model_info('meta-llama/Llama-2-7b-hf')"
```

### Monitor Training
```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Follow training log
tail -f logs/training_*.log

# Check running jobs
ps aux | grep python
```

### Kill Training
```bash
# Find process
ps aux | grep llama_finetuning

# Kill by PID
kill -9 <PID>

# Or kill all Python training processes
pkill -f llama_finetuning
```

---

## Troubleshooting

### Out of Memory
- Reduce `BATCH_SIZE` in script or pass as argument: `train_interactive.sh 4 2`
- Use fewer GPUs: `train_interactive.sh 2 4`

### Token Issues
```bash
# Verify token is set
echo $HF_TOKEN

# Test access
python -c "from huggingface_hub import model_info; info = model_info('meta-llama/Llama-2-7b-hf'); print('✓ Access granted')"
```

### Data Not Found
```bash
# Check data paths
ls -lh generation/multiwoz/converters/woz.2.2.gen/train.act.json
ls -lh generation/multiwoz/converters/woz.2.2.real/train.act.json
```

### CUDA Not Detected
```bash
# Check driver
nvidia-smi

# Check PyTorch
python -c "import torch; print(torch.__version__); print(torch.version.cuda)"

# Reinstall with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

---

## Expected Timeline

| GPUs | Stage 1 (Synthetic) | Stage 2 (Real) | Total |
|------|---------------------|----------------|-------|
| 4    | 8-12h              | 2-3h           | 10-15h |
| 2    | 16-24h             | 4-6h           | 20-30h |
| 1    | 32-48h             | 8-12h          | 40-60h |

---

## After Training

### Evaluate Model
```bash
cd eval
bash start_svr_vllm.sh
python run_inference_vllm.py
python metric.py
```

### Find Model Checkpoints
```bash
# Stage 1 output
ls -lh training_scripts/agent_sft_act_dataset.7b.2e-5.full.B4.E1.gen*

# Stage 2 output (final model)
ls -lh training_scripts/agent_sft_act_dataset.7b.2e-5.full.B4.E1.real*
```
