"""TMR detector — Oxidane/tmr-ai-text-detector (125M, RoBERTa-base)."""

import torch
from .base import Detector


class TMRDetector(Detector):
    """Pipeline-based text-classification detector. 99.28% AUROC on RAID."""

    def __init__(self) -> None:
        self._pipe = None
        self._model_id = "Oxidane/tmr-ai-text-detector"

    def load(self, device: str, fp16: bool = False) -> None:
        from transformers import pipeline

        self._pipe = pipeline(
            "text-classification",
            model=self._model_id,
            device=device,
            dtype=torch.float16 if fp16 else torch.float32,
            top_k=None,
        )

    def score(self, texts: list[str], batch_size: int = 32) -> list[float]:
        results = self._pipe(
            texts, batch_size=batch_size, truncation=True, max_length=512
        )
        probs = []
        for row in results:
            ai_prob = 0.0
            for item in row:
                if item["label"].upper() in ("AI", "LABEL_1", "1"):
                    ai_prob = item["score"]
                    break
            probs.append(ai_prob)
        return probs

    @property
    def name(self) -> str:
        return "tmr"

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def description(self) -> str:
        return "TMR 125M — best accuracy-per-param (99.28% AUROC on RAID)"

    def set_model_id(self, model_id: str) -> None:
        self._model_id = model_id
