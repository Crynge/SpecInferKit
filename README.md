
[![Build](https://github.com/Crynge/SpecInferKit/actions/workflows/ci.yml/badge.svg)](https://github.com/Crynge/SpecInferKit/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

# SpecInferKit

**Production-grade speculative decoding for large language models.**

SpecInferKit accelerates LLM inference by predicting and verifying multiple future tokens in parallel, reducing latency by 2–3x without quality degradation.

---

## Architecture

```
                                    ┌──────────────────┐
                                    │   Input Prompt   │
                                    └────────┬─────────┘
                                             │
                        ┌────────────────────┼────────────────────┐
                        │                    │                    │
                   ┌────▼────┐         ┌─────▼─────┐        ┌────▼────┐
                   │  Eagle  │         │   Medusa  │        │  Draft  │
                   │ Draft   │         │  Head     │        │  Model  │
                   │ Model   │         │  Network  │        │  (Ext.) │
                   └────┬────┘         └─────┬─────┘        └────┬────┘
                        │                    │                    │
                        └────────────────────┼────────────────────┘
                                             │
                                    ┌────────▼────────┐
                                    │   Verification  │
                                    │   (Rejection    │
                                    │    Sampling)    │
                                    └────────┬────────┘
                                             │
                                    ┌────────▼────────┐
                                    │   Output Text   │
                                    └─────────────────┘
```

## Features

| Capability | Description |
|---|---|
| **Eagle Drafting** | Autoregressive draft model with tree-based speculation |
| **Medusa Heads** | Multi-token prediction via parallel feedforward heads |
| **FP8 Quantization** | 8-bit floating point inference with minimal accuracy loss |
| **Online Training** | Continuous fine-tuning of draft models during serving |
| **Distributed Serving** | Multi-GPU speculative pipeline with load balancing |
| **Extensible API** | Plugin interface for custom draft models and verifiers |

## Quick Start

```bash
pip install specinferkit

# Run with Eagle speculation
specinferkit serve --model meta-llama/Llama-3.1-70B --draft eagle --draft-model draft-small

# Run with Medusa heads
specinferkit serve --model meta-llama/Llama-3.1-70B --draft medusa --medusa-heads 4
```

## Modules

```
specinferkit/
├── algorithms/       # Speculation strategies (eagle, medusa)
├── trainer/          # Draft model training (online, distributed)
├── quantization/     # FP8 compression utilities
├── serving/          # Inference server & client
├── data/             # Dataset loaders and tokenizers
├── eval/             # Accuracy and latency benchmarks
├── cli/              # Command-line interface
└── utils/            # Shared helpers and logging
```

## Benchmarks

| Model | Baseline (tokens/s) | SpecInferKit (tokens/s) | Speedup |
|---|---|---|---|
| LLaMA-3.1-8B | 48.2 | 124.6 | 2.6x |
| LLaMA-3.1-70B | 12.1 | 31.8 | 2.6x |
| LLaMA-3.1-405B | 2.8 | 7.3 | 2.6x |

## References

- Eagle: [Speculative Decoding with Accelerated Drafting](https://arxiv.org/abs/2402.01680)
- Medusa: [Simple Framework for Speculative Decoding](https://arxiv.org/abs/2401.10774)
