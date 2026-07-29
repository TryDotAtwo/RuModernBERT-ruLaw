import argparse
import json
from pathlib import Path

from datasets import Dataset, load_dataset

from .experiment_evidence import deduplicate_ner_rows


def parse_args():
    parser = argparse.ArgumentParser(description="Create leakage-free NER parquet splits.")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--validation-file", required=True)
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = load_dataset(
        "parquet",
        data_files={
            "train": args.train_file,
            "validation": args.validation_file,
            "test": args.test_file,
        },
    )
    rows = {split: list(dataset[split]) for split in dataset}
    deduplicated, audit = deduplicate_ner_rows(rows, seed=args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, split_rows in deduplicated.items():
        Dataset.from_list(split_rows).to_parquet(output_dir / f"{split}.parquet")
        audit[f"{split}_rows"] = len(split_rows)

    hashes = {
        split: {row["text"] for row in split_rows}
        for split, split_rows in deduplicated.items()
    }
    audit["train_validation_overlap"] = len(hashes["train"] & hashes["validation"])
    audit["train_test_overlap"] = len(hashes["train"] & hashes["test"])
    audit["validation_test_overlap"] = len(hashes["validation"] & hashes["test"])
    if any(audit[key] for key in (
        "train_validation_overlap",
        "train_test_overlap",
        "validation_test_overlap",
    )):
        raise RuntimeError("Deduplicated NER splits still overlap.")

    with (output_dir / "audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2, sort_keys=True)
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
