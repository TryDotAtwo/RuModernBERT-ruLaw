#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --job-name=mlm-multiseed

set -euo pipefail
source /mnt/pool/3/vokirova/miniforge3/bin/activate
conda activate /mnt/pool/3/vokirova/venvs/rumodernbert-pip
cd /mnt/pool/6/vokirova/rumodernbert-legal-mlm

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export XDG_CACHE_HOME=/mnt/pool/6/vokirova/rumodernbert-legal-mlm/.cache
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRITON_CACHE_DIR=/tmp/triton-mlm-$SLURM_JOB_ID
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor-mlm-$SLURM_JOB_ID
mkdir -p outputs/evidence/mlm "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

seeds=(11 23 42 67 101)
models=(base our)
paths=(deepvk/RuModernBERT-base outputs/RuModernBERT-legal-mlm-20e)

pids=()
gpu=0
for model_index in 0 1; do
  for seed in "${seeds[@]}"; do
    name="${models[$model_index]}"
    path="${paths[$model_index]}"
    CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=src python -m legal_modernbert_training.evaluate_mlm \
      --model-name-or-path "$path" \
      --tokenized-dataset-dir outputs/tokenized-sud-resh-8192 \
      --per-device-eval-batch-size 1 \
      --dataloader-num-workers 2 \
      --seed "$seed" \
      --output-dir "outputs/evidence/mlm/${name}-${seed}" \
      > "outputs/evidence/mlm/${name}-${seed}.json" 2>&1 &
    pids+=("$!")
    gpu=$(( (gpu + 1) % 8 ))
    if (( ${#pids[@]} == 8 )); then
      for pid in "${pids[@]}"; do wait "$pid"; done
      pids=()
    fi
  done
done
for pid in "${pids[@]}"; do wait "$pid"; done

PYTHONPATH=src python - <<'PY'
import json
import math
import re
from pathlib import Path

rows = []
for path in sorted(Path("outputs/evidence/mlm").glob("*.json")):
    text = path.read_text(encoding="utf-8")
    match = re.search(r'\{[\s\S]*\}\s*$', text)
    if not match:
        continue
    metrics = json.loads(match.group(0))
    model, seed = path.stem.rsplit("-", 1)
    rows.append({"model": model, "seed": int(seed), **metrics})

summary = {}
for model in ("base", "our"):
    values = [row["eval_loss"] for row in rows if row["model"] == model]
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    summary[model] = {
        "n": len(values),
        "mean_eval_loss": mean,
        "sample_std_eval_loss": math.sqrt(variance),
        "mean_perplexity": sum(math.exp(value) for value in values) / len(values),
    }
summary["relative_perplexity_improvement"] = (
    1 - summary["our"]["mean_perplexity"] / summary["base"]["mean_perplexity"]
)
output = {"runs": rows, "summary": summary}
Path("outputs/evidence/mlm/summary.json").write_text(
    json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True),
    encoding="utf-8",
)
print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
PY
