#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=320G
#SBATCH --time=00:30:00
#SBATCH --job-name=mlm-smoke
#SBATCH --output=mlm-smoke-%j.out

set -euo pipefail

source /mnt/pool/3/vokirova/miniforge3/bin/activate
conda activate /mnt/pool/3/vokirova/venvs/rumodernbert-pip
cd /mnt/pool/6/vokirova/rumodernbert-legal-mlm

export TRITON_CACHE_DIR=/tmp/triton-$SLURM_JOB_ID
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor-$SLURM_JOB_ID
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

DATALOADER_NUM_WORKERS=1 bash scripts/train_8xa100_deepspeed.sh --max-steps 2
