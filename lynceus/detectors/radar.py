"""RADAR detector — TrustSafeAI/RADAR-Vicuna-7B.

RADAR (arXiv:2307.03838) is an AI-text detector trained adversarially against a
PPO paraphraser on Vicuna-7B-v1.1 generations. Despite the name it is a ~0.4B
*RoBERTa-large* sequence classifier (``RobertaForSequenceClassification``,
24 hidden layers, hidden_size 1024) - the "Vicuna-7B" refers to the generator,
not the detector. Its value to lynceus is paraphrase robustness: AUROC ~0.92
on Vicuna-7B text *after* paraphrase attack vs ~0.65 for the OpenAI detector.

# label convention (VERIFIED from the model card's reference inference code,
# IBM/RADAR radar_examples.ipynb):

    output_probs = F.log_softmax(detector(logits), -1)[:, 0].exp()   # AI prob

so the AI-probability is the softmax of **class index 0** (LABEL_0). This differs
from many generic text-classifiers where AI = index 1. To stay robust against
model cards that do ship named labels, ``score()``:

1. if labels name AI-style classes ("AI", "AI-generated", "machine", "LLM") ->
   return that class's probability directly;
2. elif labels name Human-style classes -> return `1 - human`-prob;
3. else fall back to the RADAR convention -> return LABEL_0's probability.

# SHORT-TEXT CALIBRATION (MEASURED 2026-08-08)

Empirical probe on this exact checkpoint found RADAR **saturates toward class 0
(AI) on inputs under ~100 words**: human 1984 text scored 0.999 at 9-40 words
but 0.057 at 110 words; AI text scored ~0.99 at every length. So short human
text is a systematic false positive. ``score()`` therefore blends short-text
scores toward neutral 0.5 linearly below ``MIN_RELIABLE_WORDS`` — a short
snippet's RADAR reading is treated as weak evidence, not a confident AI verdict.

License is a Vicuna-derived **non-commercial** license (per the model card).
"""

import torch

from .base import Detector

MAX_LENGTH = 512

# Below this many words RADAR saturates toward class-0 (AI) on ALL input,
# including clearly human text. Measured threshold: reliable at >= ~100 words.
MIN_RELIABLE_WORDS = 100

# Pipeline-based text-classification, like ``tmr``. Imported at module scope so
# tests can monkeypatch a fake pipeline straight onto the instance.
from transformers import pipeline  # noqa: E402

_AI_LABELS = {
    "ai", "ai-generated", "ai generated", "ai_generated", "llm",
    "machine-generated", "machine generated", "machine-generated-text",
    "fake", "synthetic", "artificial",
}
_HUMAN_LABELS = {
    "human", "human-written", "human written", "humanwritten", "real", "gt",
    "genuine", "authored-by-human", "author",
}


def _resolve_ai_probability(row: list[dict]) -> float:
    """Map one pipeline ``('label','score')`` row to an AI probability 0-1."""
    by_label = {item["label"].strip().lower(): item["score"] for item in row}
    if not by_label:
        return 0.5

    # 1) explicit AI-named class wins.
    ai = next((v for k, v in by_label.items() if k in _AI_LABELS), None)
    if ai is not None:
        return float(ai)

    # 2) invert a named Human class.
    human = next((v for k, v in by_label.items() if k in _HUMAN_LABELS), None)
    if human is not None:
        return float(1.0 - human)

    # 3) RADAR convention: AI-probability is class index 0 (log_softmax[0]).
    if "label_0" in by_label:
        return float(by_label["label_0"])

    # 4) last resort: the largest class score.
    return float(max(v for v in by_label.values()))


def _calibrate_short_text(ai_prob: float, word_count: int) -> float:
    """Blend a RADAR score toward 0.5 for short inputs (measured saturation).

    At ``word_count >= MIN_RELIABLE_WORDS`` the raw score is returned
    unchanged. Below that, linearly interpolate toward 0.5: a 10-word snippet
    with a raw 0.999 reading becomes ~0.55 — weak evidence, not a confident
    AI verdict. The blend is monotone and never overshoots the raw score.
    """
    if word_count >= MIN_RELIABLE_WORDS or word_count <= 0:
        return float(ai_prob)
    strength = word_count / MIN_RELIABLE_WORDS
    return float(0.5 + (ai_prob - 0.5) * strength)


class RadarDetector(Detector):
    """Wrap TrustSafeAI/RADAR-Vicuna-7B as a paragraph-paraphrase-robust detector."""

    def __init__(self) -> None:
        self._pipe = None
        self._model_id = "TrustSafeAI/RADAR-Vicuna-7B"

    def load(self, device: str, fp16: bool = False) -> None:
        try:
            self._pipe = pipeline(
                "text-classification",
                model=self._model_id,
                device=device,
                dtype=torch.float16 if fp16 else torch.float32,
                top_k=None,
            )
        except Exception as exc:  # noqa: BLE001 - fatal, surface clearly
            raise RuntimeError(
                f"RADAR model '{self._model_id}' could not be loaded. "
                "It is a non-commercial (Vicuna-derived) model; if access was "
                "denied, request it on Hugging Face and log in with "
                f"huggingface-cli. Original error: {exc}"
            ) from exc

    def score(self, texts: list[str], batch_size: int = 32) -> list[float]:
        if self._pipe is None:
            raise RuntimeError("RadarDetector.load() must be called before score().")
        results = self._pipe(
            texts, batch_size=batch_size, truncation=True, max_length=MAX_LENGTH
        )
        scores = [_resolve_ai_probability(row) for row in results]
        # Short-text calibration: RADAR saturates to ~1.0 (AI) on ANY input
        # under ~100 words, so pull short readings toward neutral 0.5.
        return [
            _calibrate_short_text(score, len(text.split()))
            for score, text in zip(scores, texts)
        ]

    @property
    def name(self) -> str:
        return "radar"

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def description(self) -> str:
        return (
            "RADAR 0.4B (RoBERTa-large) - paraphrase-robust AI detector; "
            "AUROC ~0.92 post-paraphrase vs ~0.65 OpenAI; non-commercial license"
        )

    def set_model_id(self, model_id: str) -> None:
        self._model_id = model_id
        self._pipe = None
