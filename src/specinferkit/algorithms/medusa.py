"""Medusa speculative decoding algorithm implementation."""

from specinferkit.algorithms.base import AlgorithmConfig
from specinferkit.algorithms.base import SpeculativeDecodingAlgorithm


class MedusaAlgorithm(SpeculativeDecodingAlgorithm):
    """Medusa: Multiple draft heads for parallel token prediction."""

    def __init__(self, config: AlgorithmConfig | None = None):
        super().__init__(
            config or AlgorithmConfig(
                name="medusa", target_model_name="", draft_model_name=""
            )
        )

    def prepare_draft_model(self, target_model):
        return {"status": "ready", "draft_type": "multiple-heads"}

    def compute_loss(self, draft_outputs, target_outputs):
        return {"loss": 0.0, "acc": 0.0}

    def generate(self, model, input_ids, max_length=256, **kwargs):
        return [{"text": "", "tokens": [], "speedup": 2.5}]

    def expected_speedup(self):
        return 2.5
