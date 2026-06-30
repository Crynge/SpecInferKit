"""Distributed training utilities."""

import os

import torch


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def is_main_process() -> bool:
    rank = int(os.environ.get("RANK", 0))
    return rank == 0


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", 1))
