#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --job-name=eval-sud-base-local
#SBATCH --output=eval-sud-base-local-%j.out

set -euo pipefail

source /mnt/pool/3/vokirova/miniforge3/bin/activate
conda activate /mnt/pool/3/vokirova/venvs/rumodernbert-pip
cd /mnt/pool/6/vokirova/rumodernbert-legal-mlm

export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export XDG_CACHE_HOME=/mnt/pool/6/vokirova/rumodernbert-legal-mlm/.cache
export HF_HOME=/mnt/pool/6/vokirova/rumodernbert-legal-mlm/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRITON_CACHE_DIR=/tmp/triton-eval-$SLURM_JOB_ID
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor-eval-$SLURM_JOB_ID
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m legal_modernbert_training.evaluate_mlm \
  --model-name-or-path deepvk/RuModernBERT-base \
  --tokenized-dataset-dir outputs/tokenized-sud-resh-8192 \
  --per-device-eval-batch-size 1 \
  --dataloader-num-workers 8 \
  --output-dir outputs/eval-sud-base-local
