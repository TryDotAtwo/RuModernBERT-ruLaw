# RuModernBERT Legal MLM

Fixed pipeline:

- base model: `deepvk/RuModernBERT-base`
- revision: `patched-tokenizer`
- dataset: `irlspbru/RusLawOD`
- training column: `textIPS` only
- task: continued MLM pretraining
- max sequence length: `8192`
- chunk overlap: `512` tokens
- dtype: `bfloat16`
- attention: strict `flash_attention_2`
- distributed: DeepSpeed ZeRO-2
- target hardware: single node, `8xA100 40GB`

Install on cluster:

```bash
pip install -e .
```

Run:

```bash
bash scripts/train_8xa100_deepspeed.sh
```

MEPhI SLURM run:

```bash
ssh mephi
ssh basis
mkdir -p /mnt/pool/6/vokirova/rumodernbert-legal-mlm
cd /mnt/pool/6/vokirova/rumodernbert-legal-mlm
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
python -m pip install -e .
python -m pip install flash-attn --no-build-isolation
sbatch -p kaf12 scripts/mephi_start.sh
```

The code intentionally rejects non-`flash_attention_2` attention.
