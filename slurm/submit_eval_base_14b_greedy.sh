#!/bin/bash
#SBATCH --job-name=eval-base-greedy
#SBATCH --output=logs/eval_base_14b_greedy_%j.log
#SBATCH --error=logs/eval_base_14b_greedy_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=12G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpua100
#SBATCH --time=15:00:00

# ===========================================================
# Base 14B evaluation — GREEDY decoding (same params as Strong LoRA)
# Fair comparison: same decoding strategy, no LoRA
# ===========================================================

echo "=== Job started at $(date) ==="
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader)"

module purge
module load apptainer/1.4.4/gcc-15.1.0

mkdir -p $WORKDIR/mytmp_eval_14b_greedy
export APPTAINER_CACHEDIR=$WORKDIR/.apptainer_cache
export HF_HOME=$WORKDIR/.cache/huggingface

apptainer exec \
  --nv \
  --writable-tmpfs \
  --bind $WORKDIR:$WORKDIR:rw \
  --bind $WORKDIR/mytmp_eval_14b_greedy:/tmp \
  $WORKDIR/unsloth_latest.sif \
  python $WORKDIR/fine_tuning_qwen/eval_base_14b_greedy.py

echo "=== Job finished at $(date) ==="
