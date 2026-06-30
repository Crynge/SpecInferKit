"""Download utilities for datasets."""
import os

import requests
from tqdm import tqdm


DATASET_REGISTRY = {
    "tiny-gsm8k": "https://huggingface.co/datasets/gsm8k/resolve/main/test.jsonl",
    "tiny-humaneval": "https://huggingface.co/datasets/openai_humaneval/resolve/main/test.jsonl",
}


def download_dataset(name: str, output_dir: str = "./data"):
    os.makedirs(output_dir, exist_ok=True)
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(DATASET_REGISTRY.keys())}")
    url = DATASET_REGISTRY[name]
    dest = os.path.join(output_dir, f"{name}.jsonl")
    if os.path.exists(dest):
        print(f"[INFO] Dataset already exists at {dest}")
        return dest
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(desc=name, total=total, unit="B", unit_scale=True) as pbar:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))
    print(f"[INFO] Downloaded to {dest}")
    return dest
