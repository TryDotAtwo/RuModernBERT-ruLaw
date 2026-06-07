import argparse
import inspect
from dataclasses import asdict

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from .config import TrainingConfig
from .text_pipeline import build_mlm_text, chunk_token_ids, is_usable_text


def patch_flash_attn_rotary_compat() -> None:
    from flash_attn.layers.rotary import RotaryEmbedding

    signature = inspect.signature(RotaryEmbedding.__init__)
    if "pos_idx_in_fp32" in signature.parameters:
        return

    original_init = RotaryEmbedding.__init__

    def init_without_pos_idx(self, *args, pos_idx_in_fp32=None, **kwargs):
        return original_init(self, *args, **kwargs)

    RotaryEmbedding.__init__ = init_without_pos_idx


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Continued MLM pretraining for RuModernBERT on RusLawOD.")
    parser.add_argument("--output-dir", default=TrainingConfig.output_dir)
    parser.add_argument("--num-train-epochs", type=float, default=TrainingConfig.num_train_epochs)
    parser.add_argument("--learning-rate", type=float, default=TrainingConfig.learning_rate)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=TrainingConfig.gradient_accumulation_steps)
    parser.add_argument("--per-device-train-batch-size", type=int, default=TrainingConfig.per_device_train_batch_size)
    parser.add_argument("--max-steps", type=int, default=TrainingConfig.max_steps)
    parser.add_argument("--save-steps", type=int, default=TrainingConfig.save_steps)
    parser.add_argument("--dataset-files", nargs="+", default=None)
    parser.add_argument("--local_rank", "--local-rank", type=int, default=-1)
    args = parser.parse_args()

    cfg = TrainingConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        dataset_files=args.dataset_files,
    )
    cfg.validate()
    return cfg


def tokenize_dataset(dataset, tokenizer, cfg: TrainingConfig):
    def normalize(batch):
        texts = [build_mlm_text(row, cfg.text_column) for row in _rows(batch)]
        texts = [text for text in texts if is_usable_text(text, cfg.min_chars)]
        encoded = tokenizer(
            texts,
            add_special_tokens=False,
            truncation=False,
            return_special_tokens_mask=True,
        )
        input_ids = []
        special_tokens_mask = []
        for ids in encoded["input_ids"]:
            max_content_length = cfg.max_seq_length - tokenizer.num_special_tokens_to_add(pair=False)
            for chunk in chunk_token_ids(ids, max_content_length, cfg.chunk_overlap):
                with_specials = tokenizer.build_inputs_with_special_tokens(chunk)
                input_ids.append(with_specials)
                special_tokens_mask.append(tokenizer.get_special_tokens_mask(with_specials, already_has_special_tokens=True))
        return {"input_ids": input_ids, "special_tokens_mask": special_tokens_mask}

    remove_columns = dataset.column_names
    return dataset.map(
        normalize,
        batched=True,
        remove_columns=remove_columns,
        num_proc=max(1, cfg.dataloader_num_workers),
        desc="Tokenizing textIPS",
    )


def _rows(batch: dict):
    keys = list(batch.keys())
    for values in zip(*(batch[key] for key in keys)):
        yield dict(zip(keys, values))


def main() -> None:
    cfg = parse_args()

    if cfg.attn_implementation == "flash_attention_2":
        patch_flash_attn_rotary_compat()

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, revision=cfg.model_revision, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(
        cfg.model_name,
        revision=cfg.model_revision,
        attn_implementation=cfg.attn_implementation,
        torch_dtype=torch.bfloat16,
    )

    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if cfg.dataset_files:
        raw = load_dataset("parquet", data_files=cfg.dataset_files, split=cfg.train_split)
    else:
        raw = load_dataset(cfg.dataset_name, cfg.dataset_config, split=cfg.train_split)
    tokenized = tokenize_dataset(raw, tokenizer, cfg)

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=cfg.mlm_probability,
        pad_to_multiple_of=8,
    )

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        overwrite_output_dir=False,
        do_train=True,
        do_eval=False,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        num_train_epochs=cfg.num_train_epochs,
        max_steps=cfg.max_steps,
        bf16=True,
        fp16=False,
        gradient_checkpointing=cfg.gradient_checkpointing,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=3,
        dataloader_num_workers=cfg.dataloader_num_workers,
        report_to=cfg.report_to,
        seed=cfg.seed,
        remove_unused_columns=False,
        deepspeed=cfg.deepspeed_config,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=collator,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)


if __name__ == "__main__":
    main()
