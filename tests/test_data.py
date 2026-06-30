"""Tests for data pipeline."""

import pytest
from specinferkit.data.dataset import SpeculativeDataset
from specinferkit.data.streaming import StreamingCache


class TestSpeculativeDataset:
    def test_init_with_nonexistent_path(self):
        with pytest.raises(FileNotFoundError):
            SpeculativeDataset("/nonexistent/path.jsonl")

    def test_dataset_registry(self):
        from specinferkit.data.download import DATASET_REGISTRY
        assert "tiny-gsm8k" in DATASET_REGISTRY


class TestStreamingCache:
    def test_init(self):
        cache = StreamingCache(target_model=None)
        assert cache.prefetch_factor == 4
