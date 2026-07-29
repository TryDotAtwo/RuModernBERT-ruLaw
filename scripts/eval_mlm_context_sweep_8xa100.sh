#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --job-name=mlm-context-sweep

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
export TRITON_CACHE_DIR=/tmp/triton-context-$SLURM_JOB_ID
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor-context-$SLURM_JOB_ID
mkdir -p outputs/evidence/mlm-context "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

# Derive shorter, equal-cardinality views from the already frozen external 8192-token corpus.
# Preserve the final special token and its mask when truncating.
PYTHONPATH=src python - <<'PY'
from pathlib import Path
from datasets import load_from_disk

source = load_from_disk("outputs/tokenized-sud-resh-8192")
for length in (512, 2048, 8192):
    target = Path(f"outputs/tokenized-sud-resh-prefix-{length}")
    if target.exists():
        continue
    if length == 8192:
        data = source
    else:
        def truncate(batch):
            out_ids, out_masks = [], []
            for ids, mask in zip(batch["input_ids"], batch["special_tokens_mask"]):
                if len(ids) <= length:
                    out_ids.append(ids)
                    out_masks.append(mask)
                else:
                    out_ids.append(ids[: length - 1] + [ids[-1]])
                    out_masks.append(mask[: length - 1] + [mask[-1]])
            return {"input_ids": out_ids, "special_tokens_mask": out_masks}
        data = source.map(truncate, batched=True, desc=f"Truncating to {length}")
    data.save_to_disk(str(target))
    print(f"saved length={length}, rows={len(data)}, target={target}")
PY

seeds=(11 23 42 67 101)
lengths=(512 2048 8192)
models=(base law)
paths=(deepvk/RuModernBERT-base outputs/RuModernBERT-legal-mlm-20e)

pids=()
gpu=0
for length in "${lengths[@]}"; do
  for model_index in 0 1; do
    for seed in "${seeds[@]}"; do
      name="${models[$model_index]}"
      model_path="${paths[$model_index]}"
      output="outputs/evidence/mlm-context/${name}-L${length}-S${seed}.json"
      CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=src python -m legal_modernbert_training.evaluate_mlm \
        --model-name-or-path "$model_path" \
        --tokenized-dataset-dir "outputs/tokenized-sud-resh-prefix-${length}" \
        --per-device-eval-batch-size 1 \
        --dataloader-num-workers 1 \
        --seed "$seed" \
        --output-dir "outputs/evidence/mlm-context/${name}-L${length}-S${seed}" \
        > "$output" 2>&1 &
      pids+=("$!")
      gpu=$(( (gpu + 1) % 8 ))
      if (( ${#pids[@]} == 8 )); then
        for pid in "${pids[@]}"; do wait "$pid"; done
        pids=()
      fi
    done
  done
done
for pid in "${pids[@]}"; do wait "$pid"; done

PYTHONPATH=src python - <<'PY'
import json, math, re, statistics
from pathlib import Path

runs = []
pattern = re.compile(r"(base|law)-L(512|2048|8192)-S(11|23|42|67|101)$")
for path in sorted(Path("outputs/evidence/mlm-context").glob("*.json")):
    match_name = pattern.fullmatch(path.stem)
    if not match_name:
        continue
    text = path.read_text(encoding="utf-8")
    match_json = re.search(r"\{[\s\S]*\}\s*$", text)
    if not match_json:
        raise RuntimeError(f"No final JSON object in {path}")
    model, length, seed = match_name.groups()
    runs.append({"model": model, "max_length": int(length), "seed": int(seed), **json.loads(match_json.group(0))})

summary = []
tcrit_df4 = 2.7764451051977987
for length in (512, 2048, 8192):
    by_model = {name: {r["seed"]: r for r in runs if r["model"] == name and r["max_length"] == length} for name in ("base", "law")}
    seeds = sorted(set(by_model["base"]) & set(by_model["law"]))
    if len(seeds) != 5:
        raise RuntimeError(f"Expected five paired seeds at length {length}, got {seeds}")
    loss_delta = [by_model["base"][s]["eval_loss"] - by_model["law"][s]["eval_loss"] for s in seeds]
    relative_ppl = [1.0 - by_model["law"][s]["perplexity"] / by_model["base"][s]["perplexity"] for s in seeds]
    def stats(values):
        mean = statistics.mean(values)
        sd = statistics.stdev(values)
        half = tcrit_df4 * sd / math.sqrt(len(values))
        return {"mean": mean, "sample_std": sd, "ci95": [mean - half, mean + half]}
    summary.append({
        "max_length": length,
        "n_paired_seeds": len(seeds),
        "base_mean_perplexity": statistics.mean(by_model["base"][s]["perplexity"] for s in seeds),
        "law_mean_perplexity": statistics.mean(by_model["law"][s]["perplexity"] for s in seeds),
        "paired_loss_reduction": stats(loss_delta),
        "paired_relative_perplexity_reduction": stats(relative_ppl),
    })

result = {"schema_version": 1, "dataset": "lawful-good-project/sud-resh-benchmark", "protocol": "same frozen 8192-token chunks, prefix truncated with final special token preserved", "runs": runs, "summary": summary}
Path("outputs/evidence/mlm-context/summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
PY
