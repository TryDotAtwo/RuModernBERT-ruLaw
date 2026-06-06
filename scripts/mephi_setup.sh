#!/bin/bash
set -euo pipefail

PROJECT_DIR=/mnt/pool/6/vokirova/rumodernbert-legal-mlm
VENV_DIR="${PROJECT_DIR}/.venv"

mkdir -p "${PROJECT_DIR}"
cd "${PROJECT_DIR}"

python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip wheel setuptools
python -m pip install -e .
python -m pip install flash-attn --no-build-isolation
