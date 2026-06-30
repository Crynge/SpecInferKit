"""Tests for algorithm implementations."""

import pytest
from specinferkit.algorithms import get_algorithm, ALGORITHM_REGISTRY


class TestAlgorithmRegistry:
    def test_registry_contains_eagle(self):
        assert "eagle" in ALGORITHM_REGISTRY

    def test_registry_contains_medusa(self):
        assert "medusa" in ALGORITHM_REGISTRY

    def test_get_algorithm_returns_instance(self):
        algo = get_algorithm("eagle")
        assert algo is not None

    def test_get_algorithm_invalid(self):
        with pytest.raises(ValueError):
            get_algorithm("nonexistent")

    def test_eagle_expected_speedup(self):
        algo = get_algorithm("eagle")
        assert algo.expected_speedup() > 1.0

    def test_medusa_expected_speedup(self):
        algo = get_algorithm("medusa")
        assert algo.expected_speedup() > 1.0
