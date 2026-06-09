#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=320G
#SBATCH --time=48:00:00
#SBATCH --job-name=mlm-full-20e
#SBATCH --output=mlm-full-20e-%j.out

set -euo pipefail

source /mnt/pool/3/vokirova/miniforge3/bin/activate
conda activate /mnt/pool/3/vokirova/venvs/rumodernbert-pip
cd /mnt/pool/6/vokirova/rumodernbert-legal-mlm

export TRITON_CACHE_DIR=/tmp/triton-$SLURM_JOB_ID
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor-$SLURM_JOB_ID
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

export DATALOADER_NUM_WORKERS=1

bash scripts/train_8xa100_deepspeed.sh \
  --output-dir outputs/RuModernBERT-legal-mlm-20e \
  --num-train-epochs 20 \
  --per-device-train-batch-size 2 \
  --per-device-eval-batch-size 2 \
  --gradient-accumulation-steps 1 \
  --learning-rate 5e-5 \
  --save-steps 1000 \
  --eval-steps 1000 \
  --validation-ratio 0.01 \
  --early-stopping-patience 1
