import argparse
import ast
import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from .config import TrainingConfig


ENTITY_IDS = [2, 4, 9, 13, 17]
LABELS = ["O"] + [f"{prefix}-{entity_id}" for entity_id in ENTITY_IDS for prefix in ("B", "I")]
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}


def parse_args():
    parser = argparse.ArgumentParser(description="Train a ModernBERT NER/token-classification head.")
    parser.add_argument("--model-name-or-path", default="outputs/RuModernBERT-legal-mlm-20e")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--validation-file", required=True)
    parser.add_argument("--test-file", default=None)
    parser.add_argument("--output-dir", default="outputs/RuModernBERT-legal-ner")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--num-train-epochs", type=float, default=3)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--save-steps", type=int, default=5000)
    parser.add_argument("--eval-steps", type=int, default=5000)
    parser.add_argument("--logging-steps", type=int, default=100)
    parser.add_argument("--dataloader-num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=TrainingConfig.seed)
    parser.add_argument("--local_rank", "--local-rank", type=int, default=-1)
    return parser.parse_args()


def parse_spans(value: str) -> list[tuple[int, int, int]]:
    spans = []
    for item in ast.literal_eval(value):
        if len(item) >= 4 and int(item[3]) in ENTITY_IDS:
            spans.append((int(item[0]), int(item[1]), int(item[3])))
    return sorted(spans)


def label_for_offset(start: int, end: int, spans: list[tuple[int, int, int]], previous_entity: int | None) -> tuple[int, int | None]:
    if start == end:
        return -100, previous_entity
    for span_start, span_end, entity_id in spans:
        if start < span_end and end > span_start:
            prefix = "I" if previous_entity == entity_id and start > span_start else "B"
            return LABEL_TO_ID[f"{prefix}-{entity_id}"], entity_id
    return LABEL_TO_ID["O"], None


def main() -> None:
    args = parse_args()
    data_files = {"train": args.train_file, "validation": args.validation_file}
    if args.test_file:
        data_files["test"] = args.test_file

    dataset = load_dataset("csv", data_files=data_files, column_names=["text", "spans"])
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)

    def tokenize_and_align(batch):
        tokenized = tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_seq_length,
            stride=args.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
        )
        sample_mapping = tokenized.pop("overflow_to_sample_mapping")
        offsets = tokenized.pop("offset_mapping")
        labels = []
        for chunk_offsets, sample_idx in zip(offsets, sample_mapping):
            spans = parse_spans(batch["spans"][sample_idx])
            chunk_labels = []
            previous_entity = None
            for start, end in chunk_offsets:
                label_id, previous_entity = label_for_offset(start, end, spans, previous_entity)
                chunk_labels.append(label_id)
            labels.append(chunk_labels)
        tokenized["labels"] = labels
        return tokenized

    tokenized = dataset.map(
        tokenize_and_align,
        batched=True,
        remove_columns=["text", "spans"],
        num_proc=max(1, args.dataloader_num_workers),
        desc="Tokenizing NER data",
    )

    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=len(LABELS),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
        attn_implementation=TrainingConfig.attn_implementation,
        torch_dtype=torch.bfloat16,
    )
    model.gradient_checkpointing_enable()

    def compute_metrics(eval_pred):
        from seqeval.metrics import f1_score, precision_score, recall_score

        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        true_predictions = []
        true_labels = []
        for pred_row, label_row in zip(preds, labels):
            row_preds = []
            row_labels = []
            for pred_id, label_id in zip(pred_row, label_row):
                if label_id == -100:
                    continue
                row_preds.append(ID_TO_LABEL[int(pred_id)])
                row_labels.append(ID_TO_LABEL[int(label_id)])
            true_predictions.append(row_preds)
            true_labels.append(row_labels)
        return {
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions),
        }

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=False,
        do_train=True,
        do_eval=True,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=None,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        bf16=True,
        fp16=False,
        logging_steps=args.logging_steps,
        dataloader_num_workers=args.dataloader_num_workers,
        report_to="tensorboard",
        remove_unused_columns=False,
        seed=args.seed,
        deepspeed=TrainingConfig.deepspeed_config,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer, pad_to_multiple_of=8),
        compute_metrics=compute_metrics,
    )
    trainer.train()

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    with (Path(args.output_dir) / "ner_label_map.json").open("w", encoding="utf-8") as f:
        json.dump({"labels": LABELS, "label2id": LABEL_TO_ID, "id2label": ID_TO_LABEL}, f, indent=2)

    if "test" in tokenized:
        metrics = trainer.evaluate(tokenized["test"], metric_key_prefix="test")
        with (Path(args.output_dir) / "test_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
