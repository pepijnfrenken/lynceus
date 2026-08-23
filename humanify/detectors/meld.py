"""MELD detector — anon-review-meld-2026/meld (395M, Ettin-400m backbone).

MELD (arXiv:2605.06903) is the current #1 open-source AI-text detector on the
RAID leaderboard. It is a BiLM backbone (``AutoModel``, Ettin-400m) plus a
CUSTOM prototype/anchor scoring head: token embeddings are projected to a
low-rank *style space*, and each token is scored by its distance to learned
human anchors vs. learned machine-family prototypes (soft assignment via
``logsumexp`` over a scaled softmin). The document score is the mean of the
most machine-like ``rho`` fraction of tokens.

# WHY the scoring head is custom (do not "fix" this)

``AutoModel`` / ``AutoModelForSequenceClassification`` / ``pipeline()`` only
read the backbone weights and silently discard the head — every head weight
(``style_proj``, ``style_ln``, ``human_anchors``, ``family_protos``,
``family_bias``, ``log_tau``, ``op_protos``, ``op_bias``) is dropped and the
remaining randomly-initialised model returns noise. The ``Meld`` class below
is the model card's reference implementation, kept verbatim; it loads the
full checkpoint with ``load_state_dict(..., strict=True)``.

# Score semantics & thresholding

``Meld.score()`` returns ``(P(AI), raw)`` where ``raw`` is the document score
``s`` (the mean top-rho token logit) and ``P(AI) = sigmoid(s)``. P(AI) is NOT
calibrated to 0.5 — genuine human prose averages ~0.2. The 1%-FPR flag
therefore compares the RAW score against ``cfg["score_offsets"]["overall"]
["fpr_0.01"]`` (per-stratum offsets live in ``score_offsets["strata"]``).
The detector keeps those live on the instance (``threshold`` /
``strata_thresholds``) plus the raw per-unit scores from the last
``score()`` call (``last_raw_scores``) and word counts
(``last_word_counts``), see ``flag_units()``. humanify's JSON schema has no
per-detector extra-field slot, so raw/threshold stay internal and documented
here rather than in the CLI output.

# Model card warnings honored

* **>= 100 words minimum** — below ~50 words MELD flags most genuine human
  writing (measured on the model card). Like RADAR, short-text readings are
  treated as weak evidence and blended toward neutral 0.5 below
  ``MIN_RELIABLE_WORDS`` (see ``_calibrate_short_text``).
* **2048-token cap** — ``cfg["max_length"]``; longer documents must be split
  and scores aggregated. humanify's 320-word scoring-unit chunking keeps each
  unit well under the cap, so the per-unit score is the document score.
* **Keep original line breaks** — re-flowing paragraphs shifts the score; the
  model was trained on raw document formatting. Scoring units are built by
  ``prepare_scoring_units`` (the repo-wide convention every detector shares),
  which whitespace-normalises within a unit; the line-break sensitivity is
  documented here for anyone piping raw text in for reference scoring.

License is MIT (per the model card). Weights load into the Hugging Face cache
(out of git); ``HUMANIFY_MELD_DIR`` overrides the source directory (useful
fallback if the hub download is unavailable).
"""

import json
import os

import torch
import torch.nn as nn

from .base import Detector, _ensure_pad_token

# Below this many words MELD's raw reading is unreliable: per the model card,
# under ~50 words it flags most genuine human writing. Blend short-text
# readings toward neutral 0.5 (same policy as RadarDetector, whose saturation
# behaviour is analogous; see radar.py for the measured numbers).
MIN_RELIABLE_WORDS = 100

# Imported at module scope so tests can monkeypatch fakes straight onto the
# module (same pattern as editlens.py / radar.py).
from huggingface_hub import snapshot_download  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from transformers import AutoConfig, AutoModel, AutoTokenizer  # noqa: E402


class Meld(nn.Module):
    """MELD scoring model — reference implementation from the model card.

    Do not alter the scoring head: the checkpoint was trained against this
    exact architecture and every weight is load-bearing.
    """

    def __init__(self, model_dir):
        super().__init__()
        self.cfg = json.load(open(f"{model_dir}/meld_config.json"))
        r, H = self.cfg["style_rank"], self.cfg["backbone_hidden_size"]
        self.backbone = AutoModel.from_config(
            AutoConfig.from_pretrained(model_dir), attn_implementation="sdpa"
        )
        self.style_proj = nn.Linear(H, r, bias=False)
        self.style_ln = nn.LayerNorm(r)
        self.human_anchors = nn.Parameter(torch.zeros(self.cfg["n_human_anchors"], r))
        self.family_protos = nn.Parameter(torch.zeros(self.cfg["n_families"], r))
        self.family_bias = nn.Parameter(torch.zeros(self.cfg["n_families"]))
        self.log_tau = nn.Parameter(torch.zeros(()))
        self.op_protos = nn.Parameter(torch.zeros(self.cfg["n_ops"], r))
        self.op_bias = nn.Parameter(torch.zeros(self.cfg["n_ops"]))
        self.load_state_dict(load_file(f"{model_dir}/model.safetensors"), strict=True)
        self.eval()

    @torch.no_grad()
    def score(self, texts, tokenizer, device="cpu"):
        """Returns (P(AI) in [0, 1], raw document score s) per text."""
        enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                        max_length=self.cfg["max_length"],
                        return_special_tokens_mask=True).to(device)
        valid = enc["attention_mask"].bool() & ~enc["special_tokens_mask"].bool()
        h = self.backbone(input_ids=enc["input_ids"],
                          attention_mask=enc["attention_mask"]).last_hidden_state.float()
        u = self.style_ln(self.style_proj(h))                      # style coordinates
        tau = self.log_tau.clamp(-4.0, 4.0).exp()

        def sqdist(u, p):                                          # (B, L, P)
            return ((u * u).sum(-1, keepdim=True) - 2.0 * u @ p.t()
                    + (p * p).sum(-1).view(1, 1, -1))

        human = torch.logsumexp(-tau * sqdist(u, self.human_anchors), -1, keepdim=True)
        family = -tau * sqdist(u, self.family_protos) + self.family_bias.view(1, 1, -1)
        per_token = (self.cfg["tau_agg"] * torch.logsumexp(
            (family - human).clamp(-30.0, 30.0) / self.cfg["tau_agg"], dim=-1))

        # document score = mean of the most machine-like rho fraction of tokens
        x = per_token.masked_fill(~valid, torch.finfo(per_token.dtype).min)
        x, _ = x.sort(dim=1, descending=True)
        k = (valid.sum(1).clamp(min=1) * self.cfg["rho"]).ceil().clamp(min=1).long()
        keep = torch.arange(x.shape[1], device=x.device).unsqueeze(0) < k.unsqueeze(1)
        s = torch.where(keep, x, torch.zeros_like(x)).sum(1) / k.float()
        return torch.sigmoid(s).tolist(), s.tolist()


def _calibrate_short_text(ai_prob: float, word_count: int) -> float:
    """Blend a MELD score toward 0.5 for short inputs (model-card warning).

    The model card warns MELD needs >= 100 words and flags most genuine human
    writing below ~50 words. At ``word_count >= MIN_RELIABLE_WORDS`` the raw
    score is returned unchanged; below that, linearly interpolate toward
    neutral 0.5 — a short snippet's reading is weak evidence, not a confident
    AI verdict. The blend is monotone and never overshoots the raw score.
    """
    if word_count >= MIN_RELIABLE_WORDS or word_count <= 0:
        return float(ai_prob)
    strength = word_count / MIN_RELIABLE_WORDS
    return float(0.5 + (ai_prob - 0.5) * strength)


class MeldDetector(Detector):
    """Wrap anon-review-meld-2026/meld as a paragraph-AI-likeness detector."""

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._device = "cpu"
        self._model_id = "anon-review-meld-2026/meld"
        # Flagging state (see module docstring): the 1%-FPR raw-score offset
        # from meld_config.json, the per-stratum offsets, and the raw scores /
        # word counts from the most recent score() call.
        self.threshold: float | None = None
        self.strata_thresholds: dict = {}
        self.last_raw_scores: list[float] | None = None
        self.last_word_counts: list[int] | None = None

    def load(self, device: str, fp16: bool = False) -> None:
        model_dir = os.environ.get("HUMANIFY_MELD_DIR")
        try:
            if model_dir is None:
                model_dir = snapshot_download(self._model_id)
                if not os.path.isfile(os.path.join(model_dir, "meld_config.json")):
                    raise RuntimeError(
                        f"downloaded repo '{self._model_id}' is missing "
                        "meld_config.json — not a MELD checkpoint?"
                    )
            model = Meld(model_dir)
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
        except Exception as exc:  # noqa: BLE001 - any hub/dtype error is fatal
            raise RuntimeError(
                f"MELD model '{self._model_id}' could not be loaded. "
                "Set HUMANIFY_MELD_DIR to a local directory containing the "
                f"checkpoint (meld_config.json + model.safetensors) as a "
                f"fallback. Original error: {exc}"
            ) from exc

        _ensure_pad_token(tokenizer)
        if fp16:
            model = model.half()
        model.to(device).eval()
        self._model = model
        self._tokenizer = tokenizer
        self._device = device

        offsets = model.cfg.get("score_offsets", {})
        overall = offsets.get("overall", {})
        self.threshold = float(overall.get("fpr_0.01", 0.0))
        self.strata_thresholds = dict(offsets.get("strata", {}))

    def score(self, texts: list[str], batch_size: int = 32) -> list[float]:
        if self._model is None:
            raise RuntimeError("MeldDetector.load() must be called before score().")
        probs: list[float] = []
        raw: list[float] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            p, s = self._model.score(batch, self._tokenizer, self._device)
            probs.extend(p)
            raw.extend(s)
        self.last_raw_scores = [float(x) for x in raw]
        self.last_word_counts = [len(t.split()) for t in texts]
        # Short-text calibration: MELD's raw reading is unreliable under
        # ~100 words (model-card warning), so pull short readings toward 0.5.
        return [
            _calibrate_short_text(p, wc)
            for p, wc in zip(probs, self.last_word_counts)
        ]

    def flag_units(self) -> list[bool]:
        """1%-FPR flags: raw score > fpr_0.01 threshold, one per scoring unit.

        Uses the RAW document score ``s`` (not P(AI)) per the model card's
        thresholding guidance. Units under ``MIN_RELIABLE_WORDS`` words are
        unreliable for flagging (see module docstring); their raw readings
        are still compared, and ``last_word_counts`` is available for callers
        that want to gate on length.
        """
        if self.threshold is None or self.last_raw_scores is None:
            raise RuntimeError("MeldDetector.load() must be called before score().")
        return [s > self.threshold for s in self.last_raw_scores]

    @property
    def name(self) -> str:
        return "meld"

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def description(self) -> str:
        return (
            "MELD 395M (Ettin-400m) - RAID #1 open-source AI-text detector; "
            "custom anchor/prototype scoring head; MIT license"
        )

    def set_model_id(self, model_id: str) -> None:
        self._model_id = model_id
        self._model = None
        self._tokenizer = None