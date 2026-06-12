import argparse
import json
import math

import torch
from datasets import Dataset, load_dataset, load_from_disk
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)

from .config import TrainingConfig
from .text_pipeline import chunk_token_ids, is_usable_text


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a ModernBERT MLM checkpoint on an external text dataset.")
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--dataset-name", default="lawful-good-project/sud-resh-benchmark")
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-column", default="source")
    parser.add_argument("--tokenized-dataset-dir", default=None)
    parser.add_argument("--output-dir", default="outputs/mlm-eval")
    parser.add_argument("--max-seq-length", type=int, default=TrainingConfig.max_seq_length)
    parser.add_argument("--chunk-overlap", type=int, default=TrainingConfig.chunk_overlap)
    parser.add_argument("--min-chars", type=int, default=TrainingConfig.min_chars)
    parser.add_argument("--mlm-probability", type=float, default=TrainingConfig.mlm_probability)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=TrainingConfig.seed)
    parser.add_argument("--deduplicate-texts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local_rank", "--local-rank", type=int, default=-1)
    return parser.parse_args()


def tokenize_texts(dataset, tokenizer, args):
    seen = set()

    def normalize(batch):
        input_ids = []
        special_tokens_mask = []
        texts = batch[args.text_column]
        max_content_length = args.max_seq_length - tokenizer.num_special_tokens_to_add(pair=False)
        for text in texts:
            if not is_usable_text(text, args.min_chars):
                continue
            normalized = text.strip()
            if args.deduplicate_texts:
                if normalized in seen:
                    continue
                seen.add(normalized)
            encoded = tokenizer(normalized, add_special_tokens=False, truncation=False)["input_ids"]
            for chunk in chunk_token_ids(encoded, max_content_length, args.chunk_overlap):
                with_specials = tokenizer.build_inputs_with_special_tokens(chunk)
                input_ids.append(with_specials)
                special_tokens_mask.append(
                    tokenizer.get_special_tokens_mask(with_specials, already_has_special_tokens=True)
                )
        return {"input_ids": input_ids, "special_tokens_mask": special_tokens_mask}

    return dataset.map(
        normalize,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=1 if args.deduplicate_texts else max(1, args.dataloader_num_workers),
        desc=f"Tokenizing {args.text_column}",
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(
        args.model_name_or_path,
        attn_implementation=TrainingConfig.attn_implementation,
        torch_dtype=torch.bfloat16,
    )

    if args.tokenized_dataset_dir:
        eval_dataset = load_from_disk(args.tokenized_dataset_dir)
    else:
        raw = load_dataset(args.dataset_name, args.dataset_config, split=args.split)
        if args.max_samples is not None:
            raw = raw.select(range(min(args.max_samples, len(raw))))
        if args.text_column not in raw.column_names:
            raise ValueError(f"text column {args.text_column!r} not found. Available: {raw.column_names}")
        eval_dataset = tokenize_texts(raw, tokenizer, args)

    if not isinstance(eval_dataset, Dataset) or len(eval_dataset) == 0:
        raise ValueError("Evaluation dataset is empty after filtering/tokenization.")

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_probability,
        pad_to_multiple_of=8,
    )
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        do_train=False,
        do_eval=True,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        bf16=True,
        fp16=False,
        dataloader_num_workers=args.dataloader_num_workers,
        report_to="none",
        seed=args.seed,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
    )
    metrics = trainer.evaluate()
    metrics["perplexity"] = math.exp(metrics["eval_loss"]) if metrics["eval_loss"] < 20 else float("inf")
    metrics["eval_chunks"] = len(eval_dataset)
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
