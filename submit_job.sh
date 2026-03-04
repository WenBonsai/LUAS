#!/bin/bash
# Job submission script for SLURM-based clusters
# Usage: sbatch submit_job.sh

#SBATCH --job-name=luas_train
#SBATCH --output=logs/luas_%j.out
#SBATCH --error=logs/luas_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --partition=gpu

# Print job info
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# Load modules (adjust for your cluster)
# module load cuda/11.8
# module load python/3.10

# Activate environment
source .venv/bin/activate

# Set HuggingFace token (set this before submitting!)
export HF_TOKEN=${HF_TOKEN:-"your_huggingface_token_here"}
export HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}

# Check GPU availability
nvidia-smi
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

# Run training
bash train_ucloud.sh

echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
