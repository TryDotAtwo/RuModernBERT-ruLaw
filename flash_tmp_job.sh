#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --time=02:00:00
#SBATCH --job-name=flash-tmp
#SBATCH --output=flash-tmp-%j.out

set -euo pipefail

source /mnt/pool/3/vokirova/miniforge3/bin/activate

ENV_DIR=/tmp/rumodernbert-env-$SLURM_JOB_ID
conda create -y -p "$ENV_DIR" python=3.11
conda activate "$ENV_DIR"

cd /mnt/pool/6/vokirova/rumodernbert-legal-mlm

pip install --upgrade pip wheel setuptools
pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0
pip install -e .
export MAX_JOBS=1
export FLASH_ATTENTION_FORCE_BUILD=TRUE
pip install flash-attn==2.8.3 --no-build-isolation --no-cache-dir -v

python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "import flash_attn; print('flash-attn ok')"
