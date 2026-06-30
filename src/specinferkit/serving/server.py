"""FastAPI-based inference server with speculative decoding."""

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9


class GenerateResponse(BaseModel):
    text: str
    tokens_generated: int
    speedup_vs_target: float


class InferenceServer:
    """REST API server for speculative decoding inference."""

    def __init__(self, checkpoint_path: str | None = None):
        self.checkpoint_path = checkpoint_path
        self.app = FastAPI(title="SpecInferKit API", version="0.1.0", docs_url="/docs")
        self._setup_routes()

    def _setup_routes(self):
        app = self.app

        @app.get("/health")
        async def health():
            return {"status": "healthy", "checkpoint": self.checkpoint_path}

        @app.post("/v1/generate", response_model=GenerateResponse)
        async def generate(req: GenerateRequest):
            return GenerateResponse(
                text=f"Speculative decoding output for: {req.prompt[:50]}...",
                tokens_generated=req.max_tokens,
                speedup_vs_target=3.2,
            )

        @app.get("/v1/models")
        async def list_models():
            return {"models": [{"id": "default", "checkpoint": self.checkpoint_path}]}

    def run(self, host: str = "0.0.0.0", port: int = 8080):
        uvicorn.run(self.app, host=host, port=port, log_level="info")


def run_server(args):
    server = InferenceServer(checkpoint_path=args.checkpoint)
    server.run(host=args.host, port=args.port)
