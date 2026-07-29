#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --time=02:00:00
#SBATCH --job-name=build-flash
#SBATCH --output=build-flash-%j.out

set -euo pipefail

source /mnt/pool/3/vokirova/miniforge3/bin/activate
conda activate /mnt/pool/3/vokirova/venvs/rumodernbert-pip

export TMPDIR=/tmp
export PIP_CACHE_DIR=/mnt/pool/3/vokirova/pip-cache
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export MAX_JOBS=1

pip uninstall -y flash-attn || true
pip install flash-attn==2.8.3 --no-build-isolation --no-cache-dir -v

python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "import flash_attn; print('flash-attn ok')"
