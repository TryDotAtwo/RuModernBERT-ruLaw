from legal_modernbert_training.config import TrainingConfig


def test_deepspeed_config_is_enabled_by_default():
    cfg = TrainingConfig()

    assert cfg.deepspeed_config == "configs/deepspeed_zero2_a100.json"
