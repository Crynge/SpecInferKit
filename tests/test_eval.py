"""Tests for evaluation module."""

from specinferkit.eval.benchmarks import BenchmarkSuite


class TestBenchmarkSuite:
    def test_list_benchmarks(self):
        benchmarks = BenchmarkSuite.list_benchmarks()
        assert "gsm8k" in benchmarks
        assert "humaneval" in benchmarks

    def test_run_returns_dict(self):
        suite = BenchmarkSuite(benchmarks=["gsm8k", "humaneval"])
        results = suite.run(model=None, tokenizer=None)
        assert isinstance(results, dict)
        assert "gsm8k" in results
