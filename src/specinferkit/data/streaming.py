"""Streaming cache for online training mode."""

from collections.abc import Iterator
import torch


class StreamingCache:
    """On-the-fly target cache generation for online training.
    Eliminates the need for 38TB+ precomputed cache storage.
    """

    def __init__(
        self,
        target_model: object | None = None,
        prefetch_factor: int = 4,
        max_queue_size: int = 64,
        device: str = "cuda",
    ):
        self.target_model = target_model
        self.prefetch_factor = prefetch_factor
        self.max_queue_size = max_queue_size
        self.device = device
        self._queue = []

    def generate(self, input_ids: torch.Tensor) -> dict:
        """Generate target outputs on the fly."""
        with torch.no_grad():
            outputs = self.target_model(input_ids.to(self.device))
        return {
            "logits": outputs.logits.cpu(),
            "hidden_states": outputs.hidden_states[-1].cpu() if hasattr(outputs, "hidden_states") else None,
        }

    def __iter__(self) -> Iterator[dict]:
        return self

    def __next__(self) -> dict:
        if self._queue:
            return self._queue.pop(0)
        raise StopIteration
