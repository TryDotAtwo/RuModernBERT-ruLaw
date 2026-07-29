#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --job-name=base-ner-evidence

set -euo pipefail
cd /mnt/pool/6/vokirova/rumodernbert-legal-mlm

bash scripts/train_ner_8xa100.sh \
  --model-name-or-path deepvk/RuModernBERT-base \
  --train-file data/ner_hf_deduplicated/train.parquet \
  --validation-file data/ner_hf_deduplicated/validation.parquet \
  --test-file data/ner_hf_deduplicated/test.parquet \
  --output-dir outputs/RuModernBERT-base-ner-deduplicated \
  --preprocessing-num-workers 1
