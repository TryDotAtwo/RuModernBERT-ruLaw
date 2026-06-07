#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --job-name=mlm-tokenize
#SBATCH --output=mlm-tokenize-%j.out

set -euo pipefail

source /mnt/pool/3/vokirova/miniforge3/bin/activate
conda activate /mnt/pool/3/vokirova/venvs/rumodernbert-pip

cd /mnt/pool/6/vokirova/rumodernbert-legal-mlm

export TOKENIZERS_PARALLELISM=false
export HF_HOME=/mnt/pool/6/vokirova/rumodernbert-legal-mlm/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

DATA_DIR="${DATA_DIR:-/mnt/pool/6/vokirova/rumodernbert-legal-mlm/.cache/huggingface/hub/datasets--irlspbru--RusLawOD/snapshots/f850b966648499d7ff4f4bc3ef2cddb68f4ec3c0}"
DATASET_FILES=()
for i in $(seq -w 1 11); do
  DATASET_FILES+=("${DATA_DIR}/ruslawod_${i}.parquet")
done

python -m legal_modernbert_training.prepare_mlm_dataset \
  --output-dir outputs/tokenized-ruslawod-8192 \
  --dataloader-num-workers "${DATALOADER_NUM_WORKERS:-32}" \
  --dataset-files "${DATASET_FILES[@]}" \
  "$@"
