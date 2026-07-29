#!/bin/bash
#SBATCH --job-name=ruslawod-prepare
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=ruslawod-prepare-%j.out

set -euo pipefail
source /mnt/pool/3/vokirova/miniforge3/bin/activate
conda activate /mnt/pool/3/vokirova/venvs/rumodernbert-pip
cd /mnt/pool/6/vokirova/rumodernbert-legal-mlm

python - <<'PY'
from datasets import load_dataset
data_dir = "/mnt/pool/6/vokirova/rumodernbert-legal-mlm/.cache/huggingface/hub/datasets--irlspbru--RusLawOD/snapshots/f850b966648499d7ff4f4bc3ef2cddb68f4ec3c0"
files = [f"{data_dir}/ruslawod_{i:02d}.parquet" for i in range(1, 12)]
ds = load_dataset("parquet", data_files=files, split="train")
print(ds)
print("dataset prepared ok")
PY
