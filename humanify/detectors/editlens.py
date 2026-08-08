"""EditLens detector — pangram/editlens_roberta-large (0.4B, RoBERTa-large).

Scores AI-*editing* magnitude (0-1), not binary authorship probability.

EditLens (ICLR 2026, arXiv:2510.03154) trains a sequence classifier over
*classification buckets*: class 0 = "human/unedited", class n_buckets-1 =
"heavily AI-edited". The reference ``scripts/inference.py`` turns the softmax
over buckets into a continuous 0-1 *edit-magnitude* score as the expected
bucket value:

    score = (softmax(logits) @ arange(n_buckets)) / (n_buckets - 1)

Which means:

* binary variant (n_buckets=2, weights [0.0, 1.0]) reduceres to a single
  sigmoid on the logit difference -> binary F1 94.7% per KB 06;
* ternary variant (n_buckets=3, weights [0.0, 0.5, 1.0]) maps
  human -> low, "polished/edited" -> 0.5 (mid-high), "AI" -> 1.0 ->
  ternary F1 90.4%.

We replicate that score exactly. ``n_buckets`` is read from the checkpoint
config at load time, so binary and ternary checkpoints both work.

License is CC BY-NC-SA 4.0 (non-commercial) and the repo is *contact-gated*:
``load()`` therefore fails gracefully with a clear ``RuntimeError`` (never a
hard crash) when Hugging Face denies access.
"""

import torch

from .base import Detector, _ensure_pad_token

MAX_LENGTH = 512

# Imported at module scope so ``load`` can be monkeypatched in tests with fakes.
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402


class EditLensDetector(Detector):
    """Wrap pangram/editlens_roberta-large as an EditLens-detector."""

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._device = None
        self._weights = None
        self._n_buckets = 0
        self._model_id = "pangram/editlens_roberta-large"

    def load(self, device: str, fp16: bool = False) -> None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(self._model_id)
            model = AutoModelForSequenceClassification.from_pretrained(
                self._model_id
            )
        except Exception as exc:  # noqa: BLE001 - any hub/dtype error is fatal
            raise RuntimeError(
                f"EditLens model '{self._model_id}' could not be loaded. "
                "It is a *contact-gated* Hugging Face repo (license CC BY-NC-SA "
                "4.0); you must grant / be granted access (huggingface-cli login) "
                f"before use. Original error: {exc}"
            ) from exc

        _ensure_pad_token(tokenizer)

        n = int(model.config.num_labels or 2)
        self._n_buckets = max(n, 2)
        # score = expected bucket / bucket weights [0..1]
        self._weights = torch.arange(self._n_buckets, dtype=torch.float32) / (
            self._n_buckets - 1
        )

        if fp16:
            model = model.half()
            self._weights = self._weights.half()
        model.to(device).eval()
        self._model = model
        self._tokenizer = tokenizer
        self._device = device

    def score(self, texts: list[str], batch_size: int = 32) -> list[float]:
        if self._model is None:
            raise RuntimeError(
                "EditLensDetector.load() must be called before score()."
            )
        scores: list[float] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            with torch.no_grad():
                logits = self._model(
                    input_ids=enc["input_ids"].to(self._device),
                    attention_mask=enc["attention_mask"].to(self._device),
                ).logits
            probs = torch.softmax(logits.float(), dim=-1)
            weights = self._weights.float().to(probs.device)
            batch_scores = (probs @ weights).cpu().tolist()
            if isinstance(batch_scores, float):
                batch_scores = [batch_scores]
            scores.extend(batch_scores)
        return scores

    @property
    def name(self) -> str:
        return "editlens"

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def description(self) -> str:
        return (
            "EditLens 0.4B (RoBERTa-large) - AI-edit magnitude scorer; "
            "binary F1 94.7% / ternary 90.4%; CC BY-NC-SA, contact-gated"
        )

    def set_model_id(self, model_id: str) -> None:
        self._model_id = model_id
        self._model = None
        self._tokenizer = None
