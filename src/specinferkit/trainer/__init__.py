"""Training module for speculative decoding."""
from specinferkit.trainer.base import BaseTrainer
from specinferkit.trainer.online import OnlineTrainer
from specinferkit.trainer.distributed import DistributedTrainer

__all__ = ["BaseTrainer", "OnlineTrainer", "DistributedTrainer"]
