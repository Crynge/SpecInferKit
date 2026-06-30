"""Python client for the SpecInferKit inference API."""

import requests
from typing import Optional


class Client:
    """Client for interacting with SpecInferKit inference servers."""

    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url.rstrip("/")

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7) -> dict:
        resp = requests.post(
            f"{self.base_url}/v1/generate",
            json={"prompt": prompt, "max_tokens": max_tokens, "temperature": temperature},
        )
        resp.raise_for_status()
        return resp.json()

    def health(self) -> dict:
        resp = requests.get(f"{self.base_url}/health")
        resp.raise_for_status()
        return resp.json()

    def list_models(self) -> list:
        resp = requests.get(f"{self.base_url}/v1/models")
        resp.raise_for_status()
        return resp.json()["models"]
