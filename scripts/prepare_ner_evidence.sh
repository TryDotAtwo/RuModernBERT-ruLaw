#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=ner-deduplicate

set -euo pipefail
source /mnt/pool/3/vokirova/miniforge3/bin/activate
conda activate /mnt/pool/3/vokirova/venvs/rumodernbert-pip
cd /mnt/pool/6/vokirova/rumodernbert-legal-mlm

PYTHONPATH=src python -m legal_modernbert_training.prepare_deduplicated_ner \
  --train-file data/ner_hf/data/train.parquet \
  --validation-file data/ner_hf/data/validation.parquet \
  --test-file data/ner_hf/data/test.parquet \
  --output-dir data/ner_hf_deduplicated \
  --seed 42
