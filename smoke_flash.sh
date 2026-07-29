#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --job-name=smoke-flash
#SBATCH --output=smoke-flash-%j.out

set -euo pipefail

source /mnt/pool/3/vokirova/miniforge3/bin/activate
conda activate /mnt/pool/3/vokirova/venvs/rumodernbert-pip

python - <<'PY'
import torch
print("torch", torch.__version__, torch.version.cuda)
print("cuda available", torch.cuda.is_available())
print("device count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
import flash_attn
print("flash-attn ok")
PY
