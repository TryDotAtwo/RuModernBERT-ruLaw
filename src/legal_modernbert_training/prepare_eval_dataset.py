import argparse

from datasets import load_dataset
from transformers import AutoTokenizer

from .config import TrainingConfig
from .evaluate_mlm import tokenize_texts


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare tokenized MLM eval dataset without loading a model.")
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-column", default="source")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-seq-length", type=int, default=TrainingConfig.max_seq_length)
    parser.add_argument("--chunk-overlap", type=int, default=TrainingConfig.chunk_overlap)
    parser.add_argument("--min-chars", type=int, default=TrainingConfig.min_chars)
    parser.add_argument("--dataloader-num-workers", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--deduplicate-texts", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    raw = load_dataset(args.dataset_name, args.dataset_config, split=args.split)
    if args.max_samples is not None:
        raw = raw.select(range(min(args.max_samples, len(raw))))
    if args.text_column not in raw.column_names:
        raise ValueError(f"text column {args.text_column!r} not found. Available: {raw.column_names}")
    tokenized = tokenize_texts(raw, tokenizer, args)
    tokenized.save_to_disk(args.output_dir)
    print(f"saved {len(tokenized)} chunks to {args.output_dir}")


if __name__ == "__main__":
    main()
