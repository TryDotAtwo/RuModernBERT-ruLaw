#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=320G
#SBATCH --time=24:00:00
#SBATCH --job-name=legal-ner
#SBATCH --output=legal-ner-%j.out

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
export TORCHDYNAMO_DISABLE=1
export TORCHINDUCTOR_COMPILE_THREADS=1
export TRITON_CACHE_DIR=/tmp/triton-ner-$SLURM_JOB_ID
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor-ner-$SLURM_JOB_ID
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

deepspeed --num_gpus=8 --module legal_modernbert_training.train_ner_head \
  --model-name-or-path outputs/RuModernBERT-legal-mlm-20e \
  --train-file data/ner_hf/data/train.parquet \
  --validation-file data/ner_hf/data/validation.parquet \
  --test-file data/ner_hf/data/test.parquet \
  --output-dir outputs/RuModernBERT-legal-ner \
  --max-seq-length 2048 \
  --stride 256 \
  --per-device-train-batch-size 4 \
  --per-device-eval-batch-size 4 \
  --gradient-accumulation-steps 1 \
  --learning-rate 3e-5 \
  --num-train-epochs 3 \
  --save-steps 5000 \
  --eval-steps 5000 \
  --logging-steps 100 \
  --dataloader-num-workers 8 \
  "$@"
