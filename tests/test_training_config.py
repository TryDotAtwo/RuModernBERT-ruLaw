from legal_modernbert_training.config import TrainingConfig


def test_default_pipeline_is_fixed_for_legal_mlm():
    cfg = TrainingConfig()

    assert cfg.model_name == "deepvk/RuModernBERT-base"
    assert cfg.model_revision == "patched-tokenizer"
    assert cfg.dataset_name == "irlspbru/RusLawOD"
    assert cfg.dataset_files is None
    assert cfg.text_column == "textIPS"
    assert cfg.max_seq_length == 8192
    assert cfg.attn_implementation == "flash_attention_2"
    assert cfg.torch_dtype == "bfloat16"


def test_requires_flash_attention_2_strictly():
    cfg = TrainingConfig(attn_implementation="sdpa")

    try:
        cfg.validate()
    except ValueError as exc:
        assert "flash_attention_2" in str(exc)
    else:
        raise AssertionError("TrainingConfig.validate() should reject non-FA2 attention")
