#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --job-name=collect-repro

set -euo pipefail

ROOT=/mnt/pool/6/vokirova/rumodernbert-legal-mlm
ENV_DIR=/mnt/pool/3/vokirova/venvs/rumodernbert-pip
ART="$ROOT/artifacts/reproducibility"
MAX_COPY_BYTES=$((25 * 1024 * 1024))

cd "$ROOT"
source "$ENV_DIR/bin/activate"

rm -rf "$ART"
mkdir -p "$ART/environment" "$ART/logs" "$ART/metrics" "$ART/manifests"

{
  echo "generated_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "git_commit=$(git rev-parse HEAD)"
  echo "git_branch=$(git branch --show-current)"
  echo "python=$(python --version 2>&1)"
  echo "hostname=$(hostname)"
  echo "slurm_job_id=${SLURM_JOB_ID:-unknown}"
} > "$ART/environment/run_context.txt"

python -m pip freeze --all | LC_ALL=C sort > "$ART/environment/pip-freeze.txt"
python - <<'PY' > "$ART/environment/python-packages.json"
import json, platform
packages = {}
for name in ("torch", "transformers", "datasets", "deepspeed", "accelerate", "safetensors", "seqeval"):
    try:
        module = __import__(name)
        packages[name] = getattr(module, "__version__", "unknown")
    except Exception as exc:
        packages[name] = {"error": type(exc).__name__}
print(json.dumps({"python": platform.python_version(), "packages": packages}, indent=2, sort_keys=True))
PY

lscpu > "$ART/environment/lscpu.txt"
free -h > "$ART/environment/memory.txt"
nvidia-smi -L > "$ART/environment/nvidia-smi-L.txt" 2>&1 || true
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader > "$ART/environment/gpus.csv" 2>&1 || true

# Preserve scheduler accounting for every experiment mentioned in the paper/history.
for jid in 33146 33149 33152 33153; do
  sacct -j "$jid" --format=JobID,JobName,Partition,State,ExitCode,Elapsed,AllocCPUS,ReqMem,AllocTRES%80 -P \
    > "$ART/environment/sacct-$jid.txt" 2>&1 || true
done

# Compress complete SLURM stdout/stderr logs. gzip keeps even large progress-bar logs manageable.
for log in slurm-*.out; do
  [[ -f "$log" ]] || continue
  gzip -9 -c "$log" > "$ART/logs/$log.gz"
done

# Copy small text/JSON evaluation and trainer metadata while preserving paths.
while IFS= read -r -d '' file; do
  size=$(stat -c %s "$file")
  if (( size <= MAX_COPY_BYTES )); then
    dest="$ART/metrics/$file"
    mkdir -p "$(dirname "$dest")"
    cp -a "$file" "$dest"
  fi
done < <(find outputs -type f \( \
  -name '*.json' -o -name '*.jsonl' -o -name '*.csv' -o -name '*.txt' -o -name '*.log' \
\) -print0)

# Inventory every output and dataset file; hash final models and data, but do not upload them.
find outputs data -type f -printf '%s\t%TY-%Tm-%TdT%TH:%TM:%TS\t%p\n' 2>/dev/null \
  | LC_ALL=C sort -k3 > "$ART/manifests/file-inventory.tsv"

{
  find outputs -maxdepth 2 -type f \( -name '*.safetensors' -o -name '*.bin' \) -print0 2>/dev/null
  find data -maxdepth 3 -type f \( -name '*.parquet' -o -name '*.arrow' -o -name '*.json' \) -print0 2>/dev/null
} | xargs -0 -r sha256sum > "$ART/manifests/sha256-selected.txt"

# Machine-readable summary of key evidence files.
python - <<'PY' > "$ART/manifests/evidence-index.json"
import json
from pathlib import Path
root = Path("artifacts/reproducibility/metrics")
items = []
for path in sorted(root.rglob("*")):
    if path.is_file():
        items.append({"path": path.as_posix(), "bytes": path.stat().st_size})
print(json.dumps({"schema_version": 1, "files": items}, indent=2, ensure_ascii=False))
PY

# Refuse publication if common credential/private-key forms appear in collected artifacts.
if grep -RIE --binary-files=without-match \
  'BEGIN (OPENSSH|RSA|EC) PRIVATE KEY|github_pat_[A-Za-z0-9_]{20,}|ghp_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}' \
  "$ART"; then
  echo "Refusing to publish: credential-like content detected" >&2
  exit 2
fi

find "$ART" -type f -printf '%s\t%p\n' | LC_ALL=C sort -k2 > "$ART/manifests/artifact-files.tsv"
du -sh "$ART"
echo "ARTIFACT_COLLECTION_COMPLETE=$ART"
