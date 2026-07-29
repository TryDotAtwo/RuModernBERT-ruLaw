#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=legal-paper-audit

set -euo pipefail
cd /mnt/pool/6/vokirova/rumodernbert-legal-mlm

echo "=== META ==="
date --iso-8601=seconds
hostname
echo "audit_job_id=$SLURM_JOB_ID"
git rev-parse HEAD
git status --short

echo "=== HISTORICAL JOBS ==="
sacct -u vokirova \
  -S 2026-06-01 -E 2026-07-01 \
  --name=mlm-tokenize,mlm-full-20e,eval-sud-base-local,eval-sud-our-local,legal-heads,legal-ner \
  --format=JobIDRaw,JobName%26,Partition,State,ExitCode,Submit,Start,End,Elapsed,AllocTRES%60,MaxRSS,TotalCPU \
  --units=G -P || true

echo "=== ENVIRONMENT ==="
source /mnt/pool/3/vokirova/miniforge3/bin/activate
conda activate /mnt/pool/3/vokirova/venvs/rumodernbert-pip

python --version
python - <<'PY'
import json
import platform
import datasets
import deepspeed
import torch
import transformers

data = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "transformers": transformers.__version__,
    "datasets": datasets.__version__,
    "deepspeed": deepspeed.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_count": torch.cuda.device_count(),
}
if torch.cuda.is_available():
    data["devices"] = [
        {
            "index": i,
            "name": torch.cuda.get_device_name(i),
            "memory_bytes": torch.cuda.get_device_properties(i).total_memory,
        }
        for i in range(torch.cuda.device_count())
    ]
print(json.dumps(data, ensure_ascii=False, indent=2))
PY

nvidia-smi \
  --query-gpu=index,name,memory.total,driver_version \
  --format=csv,noheader || true

echo "=== JSON RESULTS ==="
find outputs -type f \
  \( -name 'trainer_state.json' \
     -o -name '*metrics*.json' \
     -o -name 'all_results.json' \
     -o -name 'label_maps.json' \
     -o -name 'ner_label_map.json' \) \
  -print0 2>/dev/null |
while IFS= read -r -d '' file; do
    echo "--- FILE: $file"
    python -m json.tool "$file" || true
done

echo "=== ARTIFACT SIZES ==="
find outputs -maxdepth 3 -type f \
  \( -name 'config.json' \
     -o -name 'model.safetensors' \
     -o -name 'pytorch_model.bin' \
     -o -name 'tokenizer.json' \) \
  -printf '%p|%s bytes|%TY-%Tm-%TdT%TH:%TM:%TS\n' 2>/dev/null |
sort

echo "=== RELEVANT LOG LINES ==="
for file in \
  mlm-tokenize-*.out \
  mlm-full-20e-*.out \
  eval-sud-base-*.out \
  eval-sud-our-*.out \
  legal-heads-*.out \
  legal-ner-*.out
do
    [ -f "$file" ] || continue
    echo "--- LOG: $file"
    grep -E \
      'eval_loss|train_loss|test_loss|precision|recall|f1|perplexity|runtime|samples_per_second|epoch|best_model_checkpoint|Traceback|Error|FAILED|Killed' \
      "$file" | tail -n 150 || true
done

echo "=== COMPLETE ==="
