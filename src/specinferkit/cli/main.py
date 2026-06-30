"""
SpecInferKit CLI — train, evaluate, serve, and manage speculative decoding models.

Usage:
  specinferkit train --config <path> [options]
  specinferkit eval --checkpoint <path> [options]
  specinferkit serve --checkpoint <path> [options]
  specinferkit download --dataset <name> [options]
  specinferkit quantize --cache-path <path> [options]
"""

import argparse


def build_parser():
    parser = argparse.ArgumentParser(
        prog="specinferkit",
        description="Production-grade speculative decoding for LLMs",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train a draft model")
    p_train.add_argument("--config", "-c", required=True, help="Path to config YAML")
    p_train.add_argument("--mode", choices=["online", "standard"], default="online")
    p_train.add_argument("--batch-size", type=int)
    p_train.add_argument("--max-steps", type=int)
    p_train.add_argument("--algorithm", choices=["eagle", "medusa", "self-spec", "dspark", "dflash"])

    p_eval = sub.add_parser("eval", help="Evaluate a trained checkpoint")
    p_eval.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    p_eval.add_argument("--benchmarks", help="Comma-separated benchmark names")
    p_eval.add_argument("--all", action="store_true", help="Run all benchmarks")
    p_eval.add_argument("--report-format", choices=["json", "html", "md"], default="json")
    p_eval.add_argument("--output", help="Output directory for reports")

    p_serve = sub.add_parser("serve", help="Start inference server")
    p_serve.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    p_serve.add_argument("--port", type=int, default=8080)
    p_serve.add_argument("--host", default="0.0.0.0")

    p_download = sub.add_parser("download", help="Download datasets")
    p_download.add_argument("--dataset", required=True)
    p_download.add_argument("--output", default="./data")

    p_quantize = sub.add_parser("quantize", help="Quantize target cache")
    p_quantize.add_argument("--cache-path", required=True)
    p_quantize.add_argument("--dtype", choices=["float8_e4m3fn", "float8_e5m2"], default="float8_e4m3fn")
    p_quantize.add_argument("--output", required=True)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        from specinferkit.trainer.cli import run_train
        run_train(args)
    elif args.command == "eval":
        from specinferkit.eval.cli import run_eval
        run_eval(args)
    elif args.command == "serve":
        from specinferkit.serving.server import run_server
        run_server(args)
    elif args.command == "download":
        from specinferkit.data.download import download_dataset
        download_dataset(args.dataset, args.output)
    elif args.command == "quantize":
        from specinferkit.quantization.fp8 import quantize_cache
        quantize_cache(args.cache_path, args.dtype, args.output)


if __name__ == "__main__":
    main()
