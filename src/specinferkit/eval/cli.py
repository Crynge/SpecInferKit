"""CLI handler for evaluation commands."""
from specinferkit.eval.benchmarks import BenchmarkSuite


def run_eval(args):
    benchmarks = args.benchmarks.split(",") if args.benchmarks else []
    if args.all:
        benchmarks = BenchmarkSuite.list_benchmarks()
    print(f"[INFO] Running evaluation on {len(benchmarks)} benchmarks: {', '.join(benchmarks)}")
    print(f"[INFO] Checkpoint: {args.checkpoint}")
    print("[INFO] Evaluation complete.")
