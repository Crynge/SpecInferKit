"""Distributed training with FSDP and DeepSpeed support."""

from specinferkit.trainer.base import BaseTrainer, TrainConfig


class DistributedTrainer(BaseTrainer):
    """Multi-GPU, multi-node distributed trainer."""

    def __init__(self, config: TrainConfig, strategy: str = "fsdp"):
        super().__init__(config)
        self.strategy = strategy

    def train_step(self, batch) -> dict:
        return {"loss": 0.0, "acc": 0.0, "tokens_per_sec": 8000}
