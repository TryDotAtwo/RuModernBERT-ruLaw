#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --time=02:00:00
#SBATCH --job-name=flash-build
#SBATCH --output=flash-build-%j.out

set -euo pipefail

source /mnt/pool/3/vokirova/miniforge3/bin/activate
conda activate /mnt/pool/3/vokirova/venvs/rumodernbert-pip

cd /mnt/pool/6/vokirova/rumodernbert-legal-mlm

export TMPDIR=/tmp
export MAX_JOBS=1
export FLASH_ATTENTION_FORCE_BUILD=TRUE

python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"

pip install --force-reinstall --no-deps flash-attn==2.8.3 --no-build-isolation --no-cache-dir -v

python -c "import flash_attn; print('flash-attn ok')"
