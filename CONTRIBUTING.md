# Contributing to SpecInferKit

We love contributions from the community! Here's how you can help make SpecInferKit better.

## Code of Conduct

By participating, you agree to our [Code of Conduct](CODE_OF_CONDUCT.md).

## How to Contribute

### Report Bugs

Open an issue at https://github.com/Crynge/SpecInferKit/issues with:
- A clear title and description
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, GPU)

### Suggest Features

Open an issue with the `enhancement` label describing:
- The problem you're solving
- Proposed solution
- Alternative approaches considered

### Submit Code

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes
4. Run tests: `pytest tests/`
5. Run lint: `ruff check src/`
6. Commit with conventional commit messages
7. Push and open a Pull Request

## Development Setup

```bash
git clone https://github.com/Crynge/SpecInferKit.git
cd SpecInferKit
pip install -e ".[dev]"
pre-commit install
```

## Commit Conventions

We follow [Conventional Commits](https://www.conventionalcommits.org/):
- `feat:` new feature
- `fix:` bug fix
- `docs:` documentation
- `test:` tests
- `refactor:` code restructuring
- `perf:` performance improvement
- `ci:` CI/CD changes

## Pull Request Guidelines

- Keep PRs focused — one feature/fix per PR
- Write tests for new functionality
- Update documentation as needed
- Add a changelog entry
- Ensure all CI checks pass

## Project Structure

```
src/specinferkit/
  cli/          # Command-line interface
  algorithms/   # Speculative decoding implementations
  data/         # Data loading and caching
  models/       # Model definitions and adapters
  trainer/      # Training loops (online + standard)
  eval/         # Evaluation benchmarks
  serving/      # REST API server
  quantization/ # FP8 quantization
  utils/        # Shared utilities
```

## Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=specinferkit

# Specific test
pytest tests/test_algorithms.py -v
```

## Questions?

Open a [Discussion](https://github.com/Crynge/SpecInferKit/discussions) or join our community channels.
