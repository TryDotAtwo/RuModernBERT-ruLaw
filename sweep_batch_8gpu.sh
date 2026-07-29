#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=320G
#SBATCH --time=03:00:00
#SBATCH --job-name=mlm-bs-sweep
#SBATCH --output=mlm-bs-sweep-%j.out

set -euo pipefail

source /mnt/pool/3/vokirova/miniforge3/bin/activate
conda activate /mnt/pool/3/vokirova/venvs/rumodernbert-pip
cd /mnt/pool/6/vokirova/rumodernbert-legal-mlm

export DATALOADER_NUM_WORKERS=1
export TRITON_CACHE_DIR=/tmp/triton-$SLURM_JOB_ID
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor-$SLURM_JOB_ID
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

for BS in 4 8 12 16; do
  echo "===== TEST BS/GPU=$BS ====="
  nvidia-smi

  if bash scripts/train_8xa100_deepspeed.sh \
    --output-dir "outputs/sweep-bs-$BS" \
    --max-steps 20 \
    --num-train-epochs 1 \
    --per-device-train-batch-size "$BS" \
    --per-device-eval-batch-size "$BS" \
    --gradient-accumulation-steps 1 \
    --save-steps 100000 \
    --eval-steps 100000 \
    --validation-ratio 0.01; then
      echo "===== BS/GPU=$BS OK ====="
      nvidia-smi
  else
      echo "===== BS/GPU=$BS FAILED ====="
      nvidia-smi
      exit 1
  fi
done
