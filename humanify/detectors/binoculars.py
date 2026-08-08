"""Binoculars detector — zero-shot perplexity ratio method."""

import math
import os
import warnings

import torch
from .base import Detector, _ensure_pad_token


class DegenerateBinocularsWarning(UserWarning):
    """Emitted when the observer and performer models are identical.

    Using the same model as both observer and performer defeats the
    cross-perplexity ratio that Binoculars relies on, so the pairing is
    degenerate and results should be treated as unreliable (KB 02 S5).
    """

MAX_LENGTH = 512
PPL_THRESHOLD = 1e-8
SIGMOID_SCALE = 10.0


def _login_hf():
    """Login to HF Hub if HF_TOKEN is set."""
    token = os.environ.get("HF_TOKEN")
    if token:
        from huggingface_hub import login
        login(token)


class BinocularsDetector(Detector):
    """Detects AI text via perplexity/cross-perplexity ratio of two LMs.

    Based on: "Spotting LLMs With Binoculars: Zero-Shot Detection of
    Machine-Generated Text" (Schwarzschild et al., ICML 2024)

    Score = log_ppl(observer) / log_cross_ppl(observer, performer).
    AI text yields scores close to 1 (both models agree); human text
    yields higher scores (observer is more surprised than performer predicts).
    We invert and sigmoid to get 0-1 AI probability.
    """

    def __init__(self) -> None:
        self._observer_model = None
        self._observer_tokenizer = None
        self._performer_model = None
        self._performer_tokenizer = None
        self._device = None
        self._observer_id = "Qwen/Qwen3.5-0.8B-Base"
        self._performer_id = "Qwen/Qwen3.5-0.8B-Base"
        # The default observer==performer pairing is degenerate (KB 02 S5).
        self._warn_if_degenerate()

    def load(self, device: str, fp16: bool = False) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        _login_hf()
        dtype = torch.float16 if fp16 else None

        self._observer_tokenizer = AutoTokenizer.from_pretrained(self._observer_id)
        _ensure_pad_token(self._observer_tokenizer)
        self._observer_model = AutoModelForCausalLM.from_pretrained(
            self._observer_id, dtype=dtype
        )
        self._observer_model.to(device).eval()

        self._performer_tokenizer = AutoTokenizer.from_pretrained(self._performer_id)
        _ensure_pad_token(self._performer_tokenizer)
        self._performer_model = AutoModelForCausalLM.from_pretrained(
            self._performer_id, dtype=dtype
        )
        self._performer_model.to(device).eval()

        self._device = device

    def score(self, texts: list[str], batch_size: int = 32) -> list[float]:
        """Score text segments using Binoculars perplexity ratio.

        Graceful degradation: when the detector is not loaded, or a segment is
        empty/whitespace-only, returns a neutral 0.5 rather than crashing.
        """
        # Not loaded yet -> neutral 0.5 for every segment.
        if self._observer_tokenizer is None:
            return [0.5 for _ in texts]

        scores = []
        for text in texts:
            if not text.strip():
                scores.append(0.5)
                continue
            enc = self._observer_tokenizer(
                text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH,
                return_token_type_ids=False,
            ).to(self._device)
            labels = enc.input_ids[:, 1:]
            L = labels.size(-1)

            with torch.no_grad():
                obs_logits = self._observer_model(**enc).logits[:, :-1]

                if self._observer_id == self._performer_id:
                    perf_logits = obs_logits
                else:
                    perf_enc = self._performer_tokenizer(
                        text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH,
                        return_token_type_ids=False,
                    ).to(self._device)
                    perf_logits = self._performer_model(**perf_enc).logits[:, :-1]

            # Log perplexity under observer
            obs_lprobs = torch.log_softmax(obs_logits, dim=-1)
            token_log_probs = obs_lprobs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            log_ppl = -token_log_probs.sum() / L

            # Cross-perplexity: performer's predictions evaluated by observer
            perf_probs = torch.softmax(perf_logits, dim=-1)
            obs_lprobs_cross = torch.log_softmax(obs_logits, dim=-1)
            cross_entropy = -(perf_probs * obs_lprobs_cross).sum(-1)
            log_cross_ppl = cross_entropy.mean()

            if log_cross_ppl.abs() < PPL_THRESHOLD:
                scores.append(0.5)
                continue

            binoc = (log_ppl / log_cross_ppl).item()

            # Normalize: binoc ≈ 1.0 for AI, > 1.0 for human
            # Sigmoid centered at 1.0, scaled
            ai_prob = 1.0 / (1.0 + math.exp((binoc - 1.0) * SIGMOID_SCALE))
            scores.append(ai_prob)

        return scores

    @property
    def is_degenerate(self) -> bool:
        """True when the observer and performer models are the same (KB 02 S5)."""
        return self._observer_id == self._performer_id

    def _warn_if_degenerate(self) -> None:
        """Emit a user warning when observer==performer makes the pair degenerate."""
        if self.is_degenerate:
            warnings.warn(
                f"Binoculars observer and performer are the same model "
                f"('{self._observer_id}') — this pair is degenerate and "
                f"cross-perplexity is meaningless. Use two different models "
                f"via set_model_id('observer,performer').",
                DegenerateBinocularsWarning,
                stacklevel=2,
            )

    @property
    def name(self) -> str:
        return "binoculars"

    @property
    def model_id(self) -> str:
        return f"{self._observer_id},{self._performer_id}"

    @property
    def description(self) -> str:
        base = "Binoculars — zero-shot perplexity ratio"
        if self.is_degenerate:
            return (
                f"{base}; degenerate pair (observer==performer: "
                f"{self._observer_id})"
            )
        return (
            f"{base} (observer: {self._observer_id}, "
            f"performer: {self._performer_id})"
        )

    def set_model_id(self, model_id: str) -> None:
        parts = [p.strip() for p in model_id.split(",")]
        if len(parts) == 1:
            self._observer_id = parts[0]
            self._performer_id = parts[0]
        elif len(parts) == 2:
            self._observer_id = parts[0]
            self._performer_id = parts[1]
        else:
            raise ValueError(
                f"binoculars --model-id expects 'model' or 'observer,performer', "
                f"got: {model_id}"
            )
        self._warn_if_degenerate()

    def get_causal_lm(self) -> tuple | None:
        """Return loaded observer model for reuse by advice generator."""
        if self._observer_model is not None:
            return (self._observer_model, self._observer_tokenizer, self._device)
        return None
