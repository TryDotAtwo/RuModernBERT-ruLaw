from dataclasses import dataclass


@dataclass
class TrainingConfig:
    model_name: str = "deepvk/RuModernBERT-base"
    model_revision: str = "patched-tokenizer"
    dataset_name: str = "irlspbru/RusLawOD"
    dataset_config: str | None = None
    dataset_files: list[str] | None = None
    train_split: str = "train"
    validation_split: str | None = None
    text_column: str = "textIPS"
    output_dir: str = "outputs/RuModernBERT-legal-mlm"
    max_seq_length: int = 8192
    chunk_overlap: int = 512
    min_chars: int = 200
    mlm_probability: float = 0.15
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-5
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    num_train_epochs: float = 1.0
    max_steps: int = -1
    max_train_samples: int | None = None
    save_steps: int = 1000
    logging_steps: int = 10
    seed: int = 42
    attn_implementation: str = "flash_attention_2"
    torch_dtype: str = "bfloat16"
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 32
    report_to: str = "tensorboard"
    deepspeed_config: str = "configs/deepspeed_zero2_a100.json"

    def validate(self) -> None:
        if self.attn_implementation != "flash_attention_2":
            raise ValueError("This training pipeline is strict: attn_implementation must be flash_attention_2.")
        if self.torch_dtype != "bfloat16":
            raise ValueError("This training pipeline is fixed to bfloat16 for A100.")
        if self.max_seq_length != 8192:
            raise ValueError("This pipeline is fixed to ModernBERT long context: max_seq_length=8192.")
        if self.chunk_overlap <= 0 or self.chunk_overlap >= self.max_seq_length:
            raise ValueError("chunk_overlap must be positive and smaller than max_seq_length.")
