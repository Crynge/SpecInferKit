#!/bin/bash
# Environment setup script for SpecInferKit

set -e

echo "=== SpecInferKit Environment Setup ==="
echo ""

# Check Python version
python_version=$(python3 --version 2>&1)
echo "Python: $python_version"

# Create virtual environment
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Created virtual environment: venv/"
fi

source venv/bin/activate

# Install package in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

echo ""
echo "=== Setup Complete ==="
echo "Run: source venv/bin/activate"
echo "Run: pytest tests/ -v to verify"
