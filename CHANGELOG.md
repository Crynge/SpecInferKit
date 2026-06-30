# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2026-07-01

### Added
- Initial public release
- Multi-algorithm speculative decoding (Eagle-3, Medusa, Self-Speculative, DSpark, DFlash)
- FP8 quantization (float8_e4m3fn, float8_e5m2) with <0.02 MAE
- Online training mode (no precomputed cache required)
- Distributed training with FSDP and DeepSpeed
- Multi-node training support
- REST API serving with load balancing
- Comprehensive evaluation suite (GSM8K, HumanEval, MT-Bench, LiveCodeBench, MATH-500)
- Docker and docker-compose setup
- GitHub Actions CI/CD
- W&B and MLflow experiment tracking
- Model export to ONNX and TorchScript
- Web dashboard for real-time monitoring
- Extensive documentation and examples
