"""Example: Using a custom model with SpecInferKit."""

from specinferkit.models.registry import ModelRegistry


def main():
    print("Supported models:")
    for model in ModelRegistry.list_supported():
        config = ModelRegistry.get_config(model)
        print(f"  - {model} ({config['params']} params)")

    model_name = "qwen3-4B"
    config = ModelRegistry.get_config(model_name)
    print(f"\nConfig for {model_name}:")
    for key, value in config.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
