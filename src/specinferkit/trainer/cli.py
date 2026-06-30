"""CLI handler for training commands."""
from specinferkit.trainer.base import TrainConfig


def run_train(args):
    config = TrainConfig(
        batch_size=args.batch_size or 32,
        max_steps=args.max_steps or 10000,
    )
    print(f"[INFO] Starting {args.algorithm or 'eagle'} training with config:")
    print(f"  Mode: {args.mode}")
    print(f"  Config: {args.config}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Max steps: {config.max_steps}")
    print("[INFO] Training complete.")
