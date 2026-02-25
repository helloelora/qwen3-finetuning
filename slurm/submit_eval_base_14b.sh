#!/bin/bash
#SBATCH --job-name=eval-base-14b
#SBATCH --output=logs/eval_base_14b_%j.log
#SBATCH --error=logs/eval_base_14b_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=12G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpua100
#SBATCH --time=15:00:00

# ==========================================
# Base model 14B evaluation (no fine-tuning)
# ==========================================

echo "=== Job started at $(date) ==="
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader)"

module purge
module load apptainer/1.4.4/gcc-15.1.0

mkdir -p $WORKDIR/mytmp_eval_14b
export APPTAINER_CACHEDIR=$WORKDIR/.apptainer_cache
export HF_HOME=$WORKDIR/.cache/huggingface

apptainer exec \
  --nv \
  --writable-tmpfs \
  --bind $WORKDIR:$WORKDIR:rw \
  --bind $WORKDIR/mytmp_eval_14b:/tmp \
  $WORKDIR/unsloth_latest.sif \
  python $WORKDIR/fine_tuning_qwen/evaluation/eval_base_14b.py

echo "=== Job finished at $(date) ==="
