"""Tests for trainer module."""

from specinferkit.trainer.base import BaseTrainer, TrainConfig
from specinferkit.trainer.online import OnlineTrainer


class TestTrainConfig:
    def test_default_values(self):
        config = TrainConfig()
        assert config.batch_size == 32
        assert config.max_steps == 10000
        assert config.learning_rate == 1e-4


class TestOnlineTrainer:
    def test_init(self):
        config = TrainConfig(batch_size=4, max_steps=10)
        trainer = OnlineTrainer(config, target_model=None, draft_model=None)
        assert trainer.config.batch_size == 4

    def test_train_step_returns_dict(self, sample_batch):
        config = TrainConfig(batch_size=4, max_steps=10)
        trainer = OnlineTrainer(config, target_model=None, draft_model=None)
        result = trainer.train_step(sample_batch)
        assert "loss" in result
        assert "acc" in result
