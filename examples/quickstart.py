"""Quick start example for SpecInferKit."""

from specinferkit.algorithms import get_algorithm
from specinferkit.trainer.base import TrainConfig
from specinferkit.trainer.online import OnlineTrainer
from specinferkit.eval.benchmarks import BenchmarkSuite


def main():
    algo = get_algorithm("eagle")
    print(f"Using algorithm: Eagle-3 (expected speedup: {algo.expected_speedup()}x)")

    config = TrainConfig(batch_size=4, max_steps=100)
    trainer = OnlineTrainer(config, target_model=None, draft_model=None)

    for step in trainer.train():
        if step % 10 == 0:
            print(f"  Step {step}/{config.max_steps}")

    suite = BenchmarkSuite(benchmarks=["gsm8k", "humaneval"])
    results = suite.run(model=None, tokenizer=None)
    print(f"Results: {results}")


if __name__ == "__main__":
    main()
