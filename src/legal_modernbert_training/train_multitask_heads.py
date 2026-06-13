import argparse
import json
import re
from collections import Counter
from pathlib import Path

import torch
from datasets import load_dataset
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments

from .config import TrainingConfig
from .text_pipeline import is_usable_text


CLASSIFIER_RE = re.compile(r"([^$\s]+)\$")


class LegalMultiTaskModel(nn.Module):
    def __init__(self, model_name: str, num_doc_types: int, num_classifier_labels: int, num_keyword_labels: int):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(
            model_name,
            config=config,
            attn_implementation=TrainingConfig.attn_implementation,
            torch_dtype=torch.bfloat16,
        )
        hidden = config.hidden_size
        self.dropout = nn.Dropout(getattr(config, "classifier_dropout", 0.1) or 0.1)
        self.doc_type_head = nn.Linear(hidden, num_doc_types)
        self.classifier_head = nn.Linear(hidden, num_classifier_labels)
        self.keywords_head = nn.Linear(hidden, num_keyword_labels)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        doc_type_labels=None,
        classifier_labels=None,
        keywords_labels=None,
    ):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(outputs.last_hidden_state[:, 0])
        doc_type_logits = self.doc_type_head(pooled)
        classifier_logits = self.classifier_head(pooled)
        keywords_logits = self.keywords_head(pooled)

        loss = None
        losses = []
        if doc_type_labels is not None:
            losses.append(nn.functional.cross_entropy(doc_type_logits.float(), doc_type_labels.long()))
        if classifier_labels is not None and classifier_logits.shape[-1] > 0:
            losses.append(nn.functional.binary_cross_entropy_with_logits(classifier_logits.float(), classifier_labels.float()))
        if keywords_labels is not None and keywords_logits.shape[-1] > 0:
            losses.append(nn.functional.binary_cross_entropy_with_logits(keywords_logits.float(), keywords_labels.float()))
        if losses:
            loss = sum(losses) / len(losses)

        return {
            "loss": loss,
            "logits": doc_type_logits,
            "doc_type_logits": doc_type_logits,
            "classifier_logits": classifier_logits,
            "keywords_logits": keywords_logits,
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Train one ModernBERT encoder with three legal metadata heads.")
    parser.add_argument("--model-name-or-path", default="outputs/RuModernBERT-legal-mlm-20e")
    parser.add_argument("--dataset-name", default=TrainingConfig.dataset_name)
    parser.add_argument("--dataset-files", nargs="+", default=None)
    parser.add_argument("--output-dir", default="outputs/RuModernBERT-legal-multitask")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--validation-ratio", type=float, default=0.01)
    parser.add_argument("--min-chars", type=int, default=TrainingConfig.min_chars)
    parser.add_argument("--classifier-min-frequency", type=int, default=25)
    parser.add_argument("--keywords-min-frequency", type=int, default=50)
    parser.add_argument("--max-classifier-labels", type=int, default=2048)
    parser.add_argument("--max-keyword-labels", type=int, default=4096)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
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


def build_input(row: dict) -> str:
    heading = row.get("headingIPS")
    text = row.get("textIPS")
    parts = [value.strip() for value in (heading, text) if isinstance(value, str) and value.strip()]
    return "\n\n".join(parts)


def parse_classifier(value) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return []
    return CLASSIFIER_RE.findall(value)


def parse_keywords(value) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return []
    return [item.strip().upper() for item in value.split(",") if item.strip()]


def keep_top(counter: Counter, min_frequency: int, max_labels: int) -> list[str]:
    labels = [label for label, count in counter.most_common() if count >= min_frequency]
    return labels[:max_labels]


def one_hot(labels: list[str], label_to_id: dict[str, int]) -> list[float]:
    values = [0.0] * len(label_to_id)
    for label in labels:
        idx = label_to_id.get(label)
        if idx is not None:
            values[idx] = 1.0
    return values


def main() -> None:
    args = parse_args()
    if args.dataset_files:
        raw = load_dataset("parquet", data_files=args.dataset_files, split="train")
    else:
        raw = load_dataset(args.dataset_name, split="train")
    if args.max_samples is not None:
        raw = raw.select(range(min(args.max_samples, len(raw))))

    doc_types = sorted({value for value in raw["doc_typeIPS"] if isinstance(value, str) and value.strip()})
    doc_type_to_id = {label: idx for idx, label in enumerate(doc_types)}

    classifier_counter = Counter()
    keyword_counter = Counter()
    for row in raw.select_columns(["classifierByIPS", "keywordsByIPS"]):
        classifier_counter.update(parse_classifier(row.get("classifierByIPS")))
        keyword_counter.update(parse_keywords(row.get("keywordsByIPS")))
    classifier_labels = keep_top(classifier_counter, args.classifier_min_frequency, args.max_classifier_labels)
    keyword_labels = keep_top(keyword_counter, args.keywords_min_frequency, args.max_keyword_labels)
    classifier_to_id = {label: idx for idx, label in enumerate(classifier_labels)}
    keyword_to_id = {label: idx for idx, label in enumerate(keyword_labels)}

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)

    def preprocess(batch):
        texts = []
        doc_type_labels = []
        classifier_targets = []
        keyword_targets = []
        for row in rows(batch):
            text = build_input(row)
            doc_type = row.get("doc_typeIPS")
            if not is_usable_text(text, args.min_chars) or doc_type not in doc_type_to_id:
                continue
            classifier_vector = one_hot(parse_classifier(row.get("classifierByIPS")), classifier_to_id)
            keyword_vector = one_hot(parse_keywords(row.get("keywordsByIPS")), keyword_to_id)
            if not any(classifier_vector) and not any(keyword_vector):
                continue
            texts.append(text)
            doc_type_labels.append(doc_type_to_id[doc_type])
            classifier_targets.append(classifier_vector)
            keyword_targets.append(keyword_vector)
        encoded = tokenizer(texts, truncation=True, max_length=args.max_seq_length)
        encoded["doc_type_labels"] = doc_type_labels
        encoded["classifier_labels"] = classifier_targets
        encoded["keywords_labels"] = keyword_targets
        return encoded

    tokenized = raw.map(
        preprocess,
        batched=True,
        remove_columns=raw.column_names,
        num_proc=max(1, args.dataloader_num_workers),
        desc="Tokenizing multitask data",
    )
    split = tokenized.train_test_split(test_size=args.validation_ratio, seed=args.seed, shuffle=True)

    model = LegalMultiTaskModel(
        args.model_name_or_path,
        num_doc_types=len(doc_types),
        num_classifier_labels=len(classifier_labels),
        num_keyword_labels=len(keyword_labels),
    )
    model.encoder.gradient_checkpointing_enable()

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
    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        data_collator=collator,
    )
    trainer.train()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "pytorch_model.bin")
    tokenizer.save_pretrained(output_dir)
    with (output_dir / "label_maps.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "doc_type": doc_type_to_id,
                "classifier": classifier_to_id,
                "keywords": keyword_to_id,
                "max_seq_length": args.max_seq_length,
            },
            f,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )


def rows(batch: dict):
    keys = list(batch.keys())
    for values in zip(*(batch[key] for key in keys)):
        yield dict(zip(keys, values))


if __name__ == "__main__":
    main()
