import argparse
import json
from pathlib import Path

import torch
from huggingface_hub import HfApi
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Package one Legal ModernBERT encoder with 4 inference heads.")
    parser.add_argument("--base-model-dir", default="outputs/RuModernBERT-legal-mlm-20e")
    parser.add_argument("--doc-heads-dir", default="outputs/RuModernBERT-legal-multitask-heads")
    parser.add_argument("--ner-head-dir", default="outputs/RuModernBERT-legal-ner")
    parser.add_argument("--output-dir", default="outputs/RuModernBERT-legal-inference")
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--commit-message", default="Upload Legal ModernBERT inference package")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_model_dir)
    doc_dir = Path(args.doc_heads_dir)
    ner_dir = Path(args.ner_head_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with (doc_dir / "label_maps.json").open("r", encoding="utf-8") as f:
        doc_maps = json.load(f)
    with (ner_dir / "ner_label_map.json").open("r", encoding="utf-8") as f:
        ner_maps = json.load(f)

    doc_state = torch.load(doc_dir / "pytorch_model.bin", map_location="cpu")
    ner_state = load_file(str(ner_dir / "model.safetensors"))

    packaged = {}
    for key, value in doc_state.items():
        if key.startswith("encoder."):
            packaged[key] = value
        elif key.startswith(("doc_type_head.", "classifier_head.", "keywords_head.")):
            packaged[key] = value
    for key, value in ner_state.items():
        if key.startswith("classifier."):
            packaged["ner_head." + key.removeprefix("classifier.")] = value

    required_prefixes = ["encoder.", "doc_type_head.", "classifier_head.", "keywords_head.", "ner_head."]
    missing = [prefix for prefix in required_prefixes if not any(key.startswith(prefix) for key in packaged)]
    if missing:
        raise RuntimeError(f"Missing packaged weights for prefixes: {missing}")

    save_file(packaged, output_dir / "model.safetensors")

    tokenizer = AutoTokenizer.from_pretrained(base_dir, use_fast=True)
    tokenizer.save_pretrained(output_dir)
    config = AutoConfig.from_pretrained(base_dir)
    config.save_pretrained(output_dir)

    heads_config = {
        "architecture": "LegalModernBertHeads",
        "max_seq_length": max(doc_maps.get("max_seq_length", 2048), 8192),
        "ner_stride": 1024,
        "doc_type_id_to_label": invert(doc_maps["doc_type"]),
        "classifier_id_to_label": invert(doc_maps["classifier"]),
        "keywords_id_to_label": invert(doc_maps["keywords"]),
        "ner_id_to_label": {str(idx): label for idx, label in enumerate(ner_maps["labels"])},
        "sources": {
            "base_model": str(base_dir),
            "doc_heads": str(doc_dir),
            "ner_head": str(ner_dir),
        },
    }
    with (output_dir / "legal_heads_config.json").open("w", encoding="utf-8") as f:
        json.dump(heads_config, f, ensure_ascii=False, indent=2, sort_keys=True)

    write_model_card(output_dir, args.repo_id)

    if args.repo_id:
        api = HfApi()
        api.create_repo(repo_id=args.repo_id, repo_type="model", exist_ok=True)
        api.upload_folder(
            repo_id=args.repo_id,
            repo_type="model",
            folder_path=str(output_dir),
            commit_message=args.commit_message,
        )
        print(f"https://huggingface.co/{args.repo_id}")
    else:
        print(output_dir)


def invert(label_to_id: dict[str, int]) -> dict[str, str]:
    return {str(idx): label for label, idx in label_to_id.items()}


def write_model_card(output_dir: Path, repo_id: str | None) -> None:
    title = repo_id or "RuModernBERT Legal Inference"
    readme = f"""---
language:
- ru
library_name: transformers
base_model: deepvk/RuModernBERT-base
datasets:
- irlspbru/RusLawOD
- lawful-good-project/sud-resh-benchmark
- TryDotAtwo/russian-legal-ner
tags:
- modernbert
- legal
- russian
- masked-lm
- token-classification
- text-classification
---

# {title}

Russian legal ModernBERT package: one shared encoder and four inference heads.

## Model

- Base encoder: [`deepvk/RuModernBERT-base`](https://huggingface.co/deepvk/RuModernBERT-base).
- Continued pretraining: MLM on [`irlspbru/RusLawOD`](https://huggingface.co/datasets/irlspbru/RusLawOD).
- Inference heads:
  - `doc_type`: document type classification from `doc_typeIPS`.
  - `classifier`: multi-label legal classifier from `classifierByIPS`.
  - `keywords`: multi-label keyword prediction from `keywordsByIPS`.
  - `ner`: token classification from [`TryDotAtwo/russian-legal-ner`](https://huggingface.co/datasets/TryDotAtwo/russian-legal-ner).

`headingIPS` and `textIPS` are used together as model input for document-level heads.

## Usage

```python
from legal_modernbert import LegalDocumentPipeline

pipe = LegalDocumentPipeline.from_pretrained("{repo_id or 'PATH_TO_MODEL'}")
result = pipe("Текст правового документа...")
```

The pipeline requires a FlashAttention 2 compatible runtime.

## Data Sources

| Source | Used for | Link |
| --- | --- | --- |
| RuModernBERT-base | Initial encoder weights | [`deepvk/RuModernBERT-base`](https://huggingface.co/deepvk/RuModernBERT-base) |
| RusLawOD | MLM, document type, classifier, keywords | [`irlspbru/RusLawOD`](https://huggingface.co/datasets/irlspbru/RusLawOD) |
| Russian legal NER | NER head | [`TryDotAtwo/russian-legal-ner`](https://huggingface.co/datasets/TryDotAtwo/russian-legal-ner) |
| Sud-resh benchmark | External MLM validation | [`lawful-good-project/sud-resh-benchmark`](https://huggingface.co/datasets/lawful-good-project/sud-resh-benchmark) |

## Benchmarks

| Evaluation | Metric | Result |
| --- | ---: | ---: |
| RusLawOD MLM validation | eval loss | 0.1337 |
| RusLawOD MLM validation | train loss | 0.1537 |
| Sud-resh MLM benchmark, this model | eval loss | 0.4473 |
| Sud-resh MLM benchmark, base `deepvk/RuModernBERT-base` | eval loss | 0.5172 |
| Sud-resh MLM benchmark | perplexity improvement vs base | ~6.8% |
| NER test | precision | 0.9970 |
| NER test | recall | 0.9884 |
| NER test | F1 | 0.9927 |
| NER test | loss | 0.00133 |
| Multitask document heads | final train loss | 0.0193 |

The NER dataset mirror keeps attribution to the original authors and source folder in its dataset card.

## Limitations

The benchmark table reports internal training/evaluation runs. Document-head quality should be validated on held-out downstream legal tasks before production use.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    main()
