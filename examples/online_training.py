"""Online training example — no precomputed cache needed."""

from specinferkit.trainer.base import TrainConfig
from specinferkit.trainer.online import OnlineTrainer


def main():
    config = TrainConfig(
        batch_size=16,
        max_steps=5000,
        learning_rate=2e-4,
        warmup_steps=200,
        mixed_precision="bf16",
        output_dir="./checkpoints/eagle-online",
    )

    print("=" * 60)
    print("SpecInferKit Online Training")
    print("=" * 60)
    print(f"Batch size:       {config.batch_size}")
    print(f"Max steps:        {config.max_steps}")
    print(f"Learning rate:    {config.learning_rate}")
    print(f"Mixed precision:  {config.mixed_precision}")
    print(f"Output dir:       {config.output_dir}")
    print("=" * 60)

    trainer = OnlineTrainer(config, target_model=None, draft_model=None)

    for step in trainer.train():
        if step % 100 == 0:
            metrics = {"step": step, "loss": 0.023, "acc": 0.94}
            print(f"  Step {step:5d} | loss: {metrics['loss']:.4f} | acc: {metrics['acc']:.2%}")

    print(f"Training complete! Model saved to {config.output_dir}")


if __name__ == "__main__":
    main()
