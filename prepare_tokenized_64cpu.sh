#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G
#SBATCH --time=08:00:00
#SBATCH --job-name=mlm-tokenize-64
#SBATCH --output=mlm-tokenize-64-%j.out

set -euo pipefail

source /mnt/pool/3/vokirova/miniforge3/bin/activate
conda activate /mnt/pool/3/vokirova/venvs/rumodernbert-pip

cd /mnt/pool/6/vokirova/rumodernbert-legal-mlm

export DATALOADER_NUM_WORKERS=64
bash scripts/prepare_tokenized_cpu.sh
