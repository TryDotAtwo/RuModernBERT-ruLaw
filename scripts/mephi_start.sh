#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00
#SBATCH --job-name=rumodernbert-legal-mlm
#SBATCH --output=slurm-%j.out

set -euo pipefail

cd /mnt/pool/6/vokirova/rumodernbert-legal-mlm

source .venv/bin/activate
bash scripts/train_8xa100_deepspeed.sh
