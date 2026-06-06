#!/usr/bin/env bash
set -euo pipefail

export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

deepspeed \
  --num_gpus=8 \
  -m legal_modernbert_training.train_mlm \
  --output-dir outputs/RuModernBERT-legal-mlm \
  --num-train-epochs 1 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 5e-5 \
  --save-steps 1000
