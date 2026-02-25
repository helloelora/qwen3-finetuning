#!/bin/bash
#SBATCH --job-name=qwen3-strong-ext
#SBATCH --output=logs/strong_lora_ext_%j.log
#SBATCH --error=logs/strong_lora_ext_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=12G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpua100
#SBATCH --time=36:00:00

# ==========================================
# Strong LoRA extended — r=64, 3 epochs
# Extended dataset (100 samples)
# ==========================================

echo "=== Job started at $(date) ==="
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader)"

module purge
module load apptainer/1.4.4/gcc-15.1.0

mkdir -p $WORKDIR/mytmp_strong_ext
export APPTAINER_CACHEDIR=$WORKDIR/.apptainer_cache
export HF_HOME=$WORKDIR/.cache/huggingface

apptainer exec \
  --nv \
  --writable-tmpfs \
  --bind $WORKDIR:$WORKDIR:rw \
  --bind $WORKDIR/mytmp_strong_ext:/tmp \
  $WORKDIR/unsloth_latest.sif \
  python $WORKDIR/fine_tuning_qwen/training/finetune_strong_lora_extended.py

echo "=== Job finished at $(date) ==="
