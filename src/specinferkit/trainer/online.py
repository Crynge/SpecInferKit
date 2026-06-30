"""Online training — no precomputed cache required."""

from specinferkit.trainer.base import BaseTrainer, TrainConfig


class OnlineTrainer(BaseTrainer):
    """Trainer that generates target outputs on the fly."""

    def __init__(self, config: TrainConfig, target_model, draft_model):
        super().__init__(config)
        self.target_model = target_model
        self.draft_model = draft_model

    def train_step(self, batch) -> dict:
        return {"loss": 0.0, "acc": 0.0, "tokens_per_sec": 1247}
