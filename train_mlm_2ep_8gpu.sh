#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00
#SBATCH --job-name=rumodernbert-mlm-2ep
#SBATCH --output=rumodernbert-mlm-2ep-%j.out

set -euo pipefail

source /mnt/pool/3/vokirova/miniforge3/bin/activate
conda activate /mnt/pool/3/vokirova/venvs/rumodernbert-pip

cd /mnt/pool/6/vokirova/rumodernbert-legal-mlm

bash scripts/train_8xa100_deepspeed.sh
