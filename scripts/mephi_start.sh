#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --job-name=rumodernbert-legal-mlm
#SBATCH --output=slurm-%j.out

set -euo pipefail

cd /mnt/pool/6/vokirova/rumodernbert-legal-mlm

source /mnt/pool/3/vokirova/miniforge3/bin/activate
conda activate /mnt/pool/3/vokirova/venvs/rumodernbert-pip

bash scripts/train_8xa100_deepspeed.sh
