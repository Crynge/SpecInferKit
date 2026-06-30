"""Algorithm registry and base classes."""
from specinferkit.algorithms.base import SpeculativeDecodingAlgorithm
from specinferkit.algorithms.eagle import EagleAlgorithm
from specinferkit.algorithms.medusa import MedusaAlgorithm

__all__ = [
    "SpeculativeDecodingAlgorithm",
    "EagleAlgorithm",
    "MedusaAlgorithm",
]

ALGORITHM_REGISTRY = {
    "eagle": EagleAlgorithm,
    "medusa": MedusaAlgorithm,
}


def get_algorithm(name: str) -> SpeculativeDecodingAlgorithm:
    if name not in ALGORITHM_REGISTRY:
        raise ValueError(
            f"Unknown algorithm '{name}'. Available: {list(ALGORITHM_REGISTRY.keys())}"
        )
    return ALGORITHM_REGISTRY[name]()
