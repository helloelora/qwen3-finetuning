#!/bin/bash
#SBATCH --job-name=qwen3-8b
#SBATCH --output=logs/8b_%j.log
#SBATCH --error=logs/8b_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=12G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpua100
#SBATCH --time=20:00:00

# ==========================================
# 8B model — Qwen3-8B, LoRA r=16, 2 epochs
# ==========================================

echo "=== Job started at $(date) ==="
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader)"

module purge
module load apptainer/1.4.4/gcc-15.1.0

mkdir -p $WORKDIR/mytmp_8b
export APPTAINER_CACHEDIR=$WORKDIR/.apptainer_cache
export HF_HOME=$WORKDIR/.cache/huggingface

apptainer exec \
  --nv \
  --writable-tmpfs \
  --bind $WORKDIR:$WORKDIR:rw \
  --bind $WORKDIR/mytmp_8b:/tmp \
  $WORKDIR/unsloth_latest.sif \
  python $WORKDIR/fine_tuning_qwen/training/finetune_8b.py

echo "=== Job finished at $(date) ==="
