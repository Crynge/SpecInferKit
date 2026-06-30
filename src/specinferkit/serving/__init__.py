"""REST API server for model serving."""

from specinferkit.serving.server import InferenceServer, run_server

__all__ = ["InferenceServer", "run_server"]
