"""Fast-DetectGPT detector — zero-shot log-probability curvature method."""

import math
import os

import torch
from .base import Detector


def _login_hf():
    """Login to HF Hub if HF_TOKEN is set."""
    token = os.environ.get("HF_TOKEN")
    if token:
        from huggingface_hub import login
        login(token)


class FastDetectDetector(Detector):
    """Conditional probability curvature via a single causal LM.

    Based on: "Fast-DetectGPT: Efficient Zero-Shot Detection of
    Machine-Generated Text via Conditional Probability Curvature" (Bao et al., 2024)
    """

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._device = None
        self._model_id = "Qwen/Qwen3.5-0.8B-Base"

    def load(self, device: str, fp16: bool = False) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        _login_hf()
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_id)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
        dtype = torch.float16 if fp16 else None
        self._model = AutoModelForCausalLM.from_pretrained(self._model_id, dtype=dtype)
        self._device = device
        self._model.to(device).eval()

    def score(self, texts: list[str], batch_size: int = 32) -> list[float]:
        """Score text segments using Fast-DetectGPT statistic."""
        scores = []
        for text in texts:
            enc = self._tokenizer(text, return_tensors="pt", truncation=True,
                                  max_length=512, return_token_type_ids=False).to(self._device)
            labels = enc.input_ids[:, 1:]
            with torch.no_grad():
                logits = self._model(**enc).logits[:, :-1]

            labels = labels.unsqueeze(-1) if labels.ndim == logits.ndim - 1 else labels
            lprobs = torch.log_softmax(logits, dim=-1)
            probs = torch.softmax(logits, dim=-1)
            log_likelihood = lprobs.gather(dim=-1, index=labels).squeeze(-1)
            mean_ref = (probs * lprobs).sum(dim=-1)
            var_ref = (probs * lprobs.square()).sum(dim=-1) - mean_ref.square()

            stat = (log_likelihood.sum(dim=-1) - mean_ref.sum(dim=-1)) / var_ref.sum(dim=-1).sqrt()
            raw = stat.item()

            # Normalize to 0-1 via sigmoid
            prob = 1.0 / (1.0 + math.exp(-raw))
            scores.append(prob)

        return scores

    @property
    def name(self) -> str:
        return "fast-detect"

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def description(self) -> str:
        return "Fast-DetectGPT — zero-shot log-prob curvature (default: Qwen3.5-0.8B)"

    def set_model_id(self, model_id: str) -> None:
        self._model_id = model_id

    def get_causal_lm(self) -> tuple | None:
        """Return loaded causal LM for reuse by advice generator."""
        if self._model is not None:
            return (self._model, self._tokenizer, self._device)
        return None