#!/bin/bash
# Run comprehensive benchmarks

set -e

BENCHMARKS=${BENCHMARKS:-"gsm8k,humaneval,mbpp,mt-bench,math500"}
CHECKPOINT=${CHECKPOINT:-"./checkpoints/latest"}
REPORT_DIR=${REPORT_DIR:-"./reports"}

echo "=== SpecInferKit Benchmark Runner ==="
echo "Benchmarks: $BENCHMARKS"
echo "Checkpoint: $CHECKPOINT"
echo "Report dir: $REPORT_DIR"
echo ""

mkdir -p "$REPORT_DIR"

specinferkit eval \
    --checkpoint "$CHECKPOINT" \
    --benchmarks "$BENCHMARKS" \
    --report-format html \
    --output "$REPORT_DIR"

echo "Benchmarks complete. Report saved to $REPORT_DIR/"
