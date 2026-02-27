#!/bin/bash
#SBATCH --job-name=qwen3-final-ans
#SBATCH --output=logs/final_answer_only_%j.log
#SBATCH --error=logs/final_answer_only_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=12G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpua100
#SBATCH --time=24:00:00

# ===========================================================
# Final Answer Only — Alex's idea
# Fine-tune ONLY on the correct answer (no reasoning at all)
# Model preserves its native <think> process
# Config: r=64, alpha=64, 3 epochs, greedy eval
# ===========================================================

echo "=== Job started at $(date) ==="
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader)"

module purge
module load apptainer/1.4.4/gcc-15.1.0

mkdir -p $WORKDIR/mytmp_final_answer_only
export APPTAINER_CACHEDIR=$WORKDIR/.apptainer_cache
export HF_HOME=$WORKDIR/.cache/huggingface

apptainer exec \
  --nv \
  --writable-tmpfs \
  --bind $WORKDIR:$WORKDIR:rw \
  --bind $WORKDIR/mytmp_final_answer_only:/tmp \
  $WORKDIR/unsloth_latest.sif \
  python $WORKDIR/fine_tuning_qwen/training/finetune_final_answer_only.py

echo "=== Job finished at $(date) ==="
