"""Dataset classes for speculative decoding training."""

import json

import torch
from torch.utils.data import Dataset



class SpeculativeDataset(Dataset):
    """Dataset for training speculative decoding draft models."""

    def __init__(
        self,
        data_path: str,
        max_length: int = 2048,
        split: str = "train",
        cache_dir: str | None = None,
    ):
        self.data_path = data_path
        self.max_length = max_length
        self.split = split
        self.cache_dir = cache_dir
        self._load_data()

    def _load_data(self):
        with open(self.data_path) as f:
            self.examples = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        input_ids = torch.tensor(example["input_ids"][: self.max_length], dtype=torch.long)
        labels = torch.tensor(example["labels"][: self.max_length], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": torch.ones_like(input_ids),
        }
