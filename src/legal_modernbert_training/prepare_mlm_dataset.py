import argparse

from datasets import load_dataset
from transformers import AutoTokenizer

from .config import TrainingConfig
from .train_mlm import tokenize_dataset


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Prepare tokenized MLM dataset for RuModernBERT.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-files", nargs="+", required=True)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--dataloader-num-workers", type=int, default=TrainingConfig.dataloader_num_workers)
    args = parser.parse_args()

    cfg = TrainingConfig(
        dataset_files=args.dataset_files,
        max_train_samples=args.max_train_samples,
        dataloader_num_workers=args.dataloader_num_workers,
    )
    cfg.validate()
    cfg.output_dir = args.output_dir
    return cfg


def main() -> None:
    cfg = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, revision=cfg.model_revision, use_fast=True)
    raw = load_dataset("parquet", data_files=cfg.dataset_files, split=cfg.train_split)
    if cfg.max_train_samples is not None:
        raw = raw.select(range(min(cfg.max_train_samples, len(raw))))
    tokenized = tokenize_dataset(raw, tokenizer, cfg)
    tokenized.save_to_disk(cfg.output_dir)
    print(f"saved tokenized dataset to {cfg.output_dir}")


if __name__ == "__main__":
    main()
