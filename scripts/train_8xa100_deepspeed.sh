#!/usr/bin/env bash
set -euo pipefail

export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/mnt/pool/6/vokirova/rumodernbert-legal-mlm/.cache/triton}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/mnt/pool/6/vokirova/rumodernbert-legal-mlm/.cache}"
export HF_HOME="${HF_HOME:-/mnt/pool/6/vokirova/rumodernbert-legal-mlm/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

DATA_DIR="${DATA_DIR:-/mnt/pool/6/vokirova/rumodernbert-legal-mlm/.cache/huggingface/hub/datasets--irlspbru--RusLawOD/snapshots/f850b966648499d7ff4f4bc3ef2cddb68f4ec3c0}"
DATASET_FILES=()
for i in $(seq -w 1 11); do
  DATASET_FILES+=("${DATA_DIR}/ruslawod_${i}.parquet")
done

deepspeed \
  --num_gpus=8 \
  --module legal_modernbert_training.train_mlm \
  --output-dir outputs/RuModernBERT-legal-mlm \
  --num-train-epochs 2 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 5e-5 \
  --save-steps 1000 \
  --dataset-files "${DATASET_FILES[@]}" \
  "$@"
