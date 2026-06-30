"""Pytest configuration and shared fixtures."""

import pytest


@pytest.fixture
def sample_config():
    return {
        "algorithm": "eagle",
        "target_model": {"name": "Qwen3-0.5B", "dtype": "bfloat16"},
        "draft_model": {"name": "Qwen3-0.5B-Draft", "trainable": True},
        "training": {"batch_size": 4, "max_steps": 10, "mode": "online"},
    }


@pytest.fixture
def sample_batch():
    import torch
    return {
        "input_ids": torch.randint(0, 100, (2, 128)),
        "labels": torch.randint(0, 100, (2, 128)),
        "attention_mask": torch.ones(2, 128),
    }
