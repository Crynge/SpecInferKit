"""Central model registry for supported architectures."""

SUPPORTED_MODELS = {
    "qwen3": {
        "0.5B": {"params": "0.5B", "default_dtype": "bfloat16"},
        "4B": {"params": "4B", "default_dtype": "bfloat16"},
        "8B": {"params": "8B", "default_dtype": "bfloat16"},
        "14B": {"params": "14B", "default_dtype": "bfloat16"},
    },
    "gemma4": {
        "12B": {"params": "12B", "default_dtype": "bfloat16"},
    },
    "llama3": {
        "8B": {"params": "8B", "default_dtype": "bfloat16"},
        "70B": {"params": "70B", "default_dtype": "bfloat16"},
    },
}


class ModelRegistry:
    """Registry for supported model architectures and their configurations."""

    @classmethod
    def list_supported(cls) -> list:
        result = []
        for family, variants in SUPPORTED_MODELS.items():
            for size in variants:
                result.append(f"{family}-{size}")
        return result

    @classmethod
    def get_config(cls, model_name: str) -> dict:
        for family, variants in SUPPORTED_MODELS.items():
            for size, _config in variants.items():
                if f"{family}-{size}" == model_name:
                    return {"family": family, "size": size, **_config}
        raise ValueError(f"Unknown model: {model_name}. Supported: {cls.list_supported()}")
