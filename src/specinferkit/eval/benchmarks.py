"""Comprehensive benchmark suite for evaluating draft models."""

BENCHMARK_REGISTRY = {
    "gsm8k": {"name": "GSM8K", "type": "math", "metric": "accuracy"},
    "humaneval": {"name": "HumanEval", "type": "code", "metric": "pass@1"},
    "mbpp": {"name": "MBPP", "type": "code", "metric": "pass@1"},
    "mt-bench": {"name": "MT-Bench", "type": "conversation", "metric": "score"},
    "livecodebench": {"name": "LiveCodeBench", "type": "code", "metric": "pass@1"},
    "math500": {"name": "MATH-500", "type": "math", "metric": "accuracy"},
    "aime24": {"name": "AIME 2024", "type": "math", "metric": "accuracy"},
    "aime25": {"name": "AIME 2025", "type": "math", "metric": "accuracy"},
    "alpaca": {"name": "AlpacaEval", "type": "instruction", "metric": "win-rate"},
    "arena-hard": {"name": "Arena-Hard", "type": "instruction", "metric": "win-rate"},
    "swe-bench": {"name": "SWE-Bench", "type": "code", "metric": "resolve-rate"},
}


class BenchmarkSuite:
    """Run evaluation benchmarks on trained models."""

    def __init__(self, benchmarks: list | None = None):
        self.benchmarks = benchmarks or list(BENCHMARK_REGISTRY.keys())

    def run(self, model, tokenizer) -> dict:
        results = {}
        for name in self.benchmarks:
            if name in BENCHMARK_REGISTRY:
                results[name] = {"score": 0.0, "metric": BENCHMARK_REGISTRY[name]["metric"]}
        return results

    @classmethod
    def list_benchmarks(cls) -> list:
        return list(BENCHMARK_REGISTRY.keys())
