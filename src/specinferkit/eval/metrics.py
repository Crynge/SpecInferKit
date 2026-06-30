"""Metrics collection and reporting."""

import json
from typing import Dict


class MetricsCollector:
    """Collect and format evaluation metrics."""

    def __init__(self):
        self.metrics = {}

    def add(self, name: str, value: float, metadata: dict = None):
        self.metrics[name] = {"value": value, "metadata": metadata or {}}

    def to_dict(self) -> dict:
        return self.metrics

    def to_json(self, path: str):
        with open(path, "w") as f:
            json.dump(self.metrics, f, indent=2)

    def generate_report(self, format: str = "md") -> str:
        if format == "md":
            lines = ["# Evaluation Report\n"]
            for name, data in self.metrics.items():
                lines.append(f"- **{name}**: {data['value']}")
            return "\n".join(lines)
        return json.dumps(self.metrics, indent=2)
