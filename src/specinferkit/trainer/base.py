"""Base trainer with shared training logic."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainConfig:
    batch_size: int = 32
    max_steps: int = 10000
    learning_rate: float = 1e-4
    warmup_steps: int = 500
    grad_accumulation: int = 4
    mixed_precision: str = "bf16"
    log_interval: int = 10
    save_interval: int = 1000
    eval_interval: int = 500
    output_dir: str = "./checkpoints"


class BaseTrainer(ABC):
    """Abstract base trainer with shared logic."""

    def __init__(self, config: TrainConfig):
        self.config = config
        self.global_step = 0

    @abstractmethod
    def train_step(self, batch) -> dict:
        """Execute a single training step."""

    def train(self):
        """Main training loop."""
        while self.global_step < self.config.max_steps:
            yield self.global_step
            self.global_step += 1
