"""Base class for speculative decoding algorithms."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class AlgorithmConfig:
    name: str
    target_model_name: str
    draft_model_name: str
    learning_rate: float = 1e-4
    batch_size: int = 32
    max_steps: int = 10000
    mixed_precision: str = "bf16"
    gradient_checkpointing: bool = False


class SpeculativeDecodingAlgorithm(ABC):
    """Base class that all speculative decoding algorithms must implement."""

    def __init__(self, config: Optional[AlgorithmConfig] = None):
        self.config = config or AlgorithmConfig(
            name=self.__class__.__name__,
            target_model_name="",
            draft_model_name="",
        )

    @abstractmethod
    def prepare_draft_model(self, target_model) -> object:
        """Prepare the draft model based on the target model."""

    @abstractmethod
    def compute_loss(self, draft_outputs, target_outputs) -> dict:
        """Compute training loss for the draft model."""

    @abstractmethod
    def generate(self, model, input_ids, max_length: int, **kwargs) -> list:
        """Generate tokens using speculative decoding."""

    @abstractmethod
    def expected_speedup(self) -> float:
        """Return the expected speedup ratio."""
