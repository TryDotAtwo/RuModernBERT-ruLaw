#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --time=00:30:00
#SBATCH --job-name=modernbert-smoke
#SBATCH --output=modernbert-smoke-%j.out

set -euo pipefail

source /mnt/pool/3/vokirova/miniforge3/bin/activate
conda activate /mnt/pool/3/vokirova/venvs/rumodernbert-pip
export TRITON_CACHE_DIR=/mnt/pool/6/vokirova/rumodernbert-legal-mlm/.cache/triton
export XDG_CACHE_HOME=/mnt/pool/6/vokirova/rumodernbert-legal-mlm/.cache
export HF_HOME=/mnt/pool/6/vokirova/rumodernbert-legal-mlm/.cache/huggingface
mkdir -p "$TRITON_CACHE_DIR" "$HF_HOME"

cd /mnt/pool/6/vokirova/rumodernbert-legal-mlm

deepspeed --num_gpus=8 --module legal_modernbert_training.train_mlm \
  --output-dir outputs/smoke \
  --max-steps 2 \
  --save-steps 1000
