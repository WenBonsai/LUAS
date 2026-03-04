#!/bin/bash
# Job submission script for UCloud (run.ai style)
# Usage: bash submit_ucloud_job.sh

set -e

JOB_NAME="luas-training-$(date +%Y%m%d-%H%M%S)"
NUM_GPUS=${NUM_GPUS:-4}

echo "=========================================="
echo "Submitting UCloud Job"
echo "Job Name: $JOB_NAME"
echo "GPUs: $NUM_GPUS"
echo "=========================================="

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN is not set!"
    echo "Please set it first: export HF_TOKEN=your_token_here"
    exit 1
fi

# Create logs directory
mkdir -p logs

# Submit job (adjust command based on UCloud CLI)
# If using direct execution:
echo "Starting training in background..."
nohup bash train_ucloud.sh > logs/${JOB_NAME}.log 2>&1 &

JOB_PID=$!
echo "Job started with PID: $JOB_PID"
echo "Monitor with: tail -f logs/${JOB_NAME}.log"
echo "Check status: ps -p $JOB_PID"
echo "Kill job: kill $JOB_PID"

# Save job info
echo "${JOB_PID}" > logs/${JOB_NAME}.pid
echo "Job info saved to logs/${JOB_NAME}.pid"
