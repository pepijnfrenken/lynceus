"""Base class for all detection methods."""

from abc import ABC, abstractmethod


class Detector(ABC):
    """Pluggable AI text detector.

    Each detector loads model(s) in `load()` and returns per-sentence
    AI probabilities (0-1, higher = more AI-like) from `score()`.
    """

    @abstractmethod
    def load(self, device: str, fp16: bool = False) -> None:
        """Load model(s) onto device. Called once at startup."""

    @abstractmethod
    def score(self, sentences: list[str], batch_size: int = 32) -> list[float]:
        """Score sentences for AI-likeness. Returns probabilities 0-1."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short display name (e.g. 'tmr', 'binoculars')."""

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Primary HuggingFace model ID."""

    @property
    def description(self) -> str:
        """One-line description for --help."""
        return ""

    def set_model_id(self, model_id: str) -> None:
        """Override the default model ID. Not all detectors support this."""
        raise NotImplementedError(f"Detector '{self.name}' does not support --model-id override.")
