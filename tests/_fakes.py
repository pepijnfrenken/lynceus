"""Test-only fake detectors and small synthetic AI/human corpora.

No model downloads here -- ``FakeDetector`` returns deterministic scores based
on the text's marker vocabulary, so smoke tests exercise the full harness
(load -> score -> per-layer metrics -> JSON) without touching weights.
"""

_AI_MARKERS = ("furthermore", "moreover", "in conclusion", "it is important",
               "robust", "facilitate", "significant", "holistic",
               "comprehensive", "consequently", "myriad", "pivotal",
               "in today", "crucial", "leverage", "substantial")


def aiish(text: str) -> bool:
    t = text.lower()
    return any(m in t for m in _AI_MARKERS)


class FakeDetector:
    """Deterministic stub: 0.8 for AI-ish text, 0.2 otherwise."""

    def __init__(self, name: str):
        self.name = name
        self._model_id = f"fake/{name}"

    def load(self, device="cpu", fp16=False):
        return None

    def score(self, texts, batch_size=16):
        return [0.8 if aiish(t) else 0.2 for t in texts]

    @property
    def model_id(self):
        return self._model_id


HUMAN_TEXTS = [
    "We drove up the old mountain road before dawn, past the cabin that used "
    "to sell jams. The mist sat on the lake and the woodsmoke hung about. "
    "My uncle put his coffee down on the log and did not say a thing.",
    "The kids made a fort out of the sofa cushions again and claimed the "
    "living room. The dog settled in the middle of it all and would not be "
    "moved. Everyone left her there.",
    "I learnt to mend a flat by watching the mechanic twice. Now the spare "
    "fits in the boot and the jack is where I left it. It usually holds to "
    "the end of the lane.",
    "Rain came down hard and flattened the garden, and the rhubarb took it "
    "best. The cloche stayed off overnight by accident, so the lettuces "
    "raised a fuss all the next week. We replanted and moved on.",
]

AI_TEXTS = [
    "In conclusion, this approach significantly enhances the overall "
    "efficiency of the process. Moreover, it facilitates seamless "
    "integration across multiple domains. Furthermore, such a robust "
    "framework is crucial for maintaining long-term scalability.",
    "It is important to note that leveraging modern tools fosters "
    "innovation. Additionally, navigating the evolving landscape paves the "
    "way for significant improvements. In summary, the achieved synergy is "
    "robust and indispensable.",
    "A comprehensive framework is pivotal to address the challenges at "
    "hand. Not only does it streamline operations, but it also unlocks "
    "myriad opportunities. Consequently, stakeholders can achieve "
    "substantial and measurable progress.",
    "In today's world, it is crucial to embrace a holistic perspective. "
    "By fostering collaboration and leveraging cutting-edge solutions, "
    "we facilitate sustainable and robust development. Moreover, "
    "transparency remains a pivotal factor in this process.",
]


def synthetic_corpus(n_human=4, n_ai=4):
    texts = []
    labels = []
    for h in HUMAN_TEXTS[:n_human]:
        texts.append(h)
        labels.append(0.0)
    for a in AI_TEXTS[:n_ai]:
        texts.append(a)
        labels.append(1.0)
    return texts, labels


def fake_specs(names=("tmr", "desklib", "fast-detect", "binoculars(degen)")):
    """Return a list of (name, FakeDetector) specs the harness can use."""
    return [(n, FakeDetector(n)) for n in names]


class FakeClassifierOutput:
    """Mimics transformers.ModelOutput with a .logits tensor."""

    def __init__(self, logits):
        self.logits = logits


class FakeSeqClassifier:
    """Dummy AutoModelForSequenceClassification replacement (no weights).

    Forward pass returns ``FakeClassifierOutput`` of stacked logits for each
    input. n_buckets is inferred from ``config.num_labels`` (default 2).
    Methods are no-ops so ``load()`` (eval/device/dtype) works.
    """

    def __init__(self, num_labels=2):
        class Cfg:
            pass
        self.config = Cfg()
        self.config.num_labels = num_labels
        self._device = "cpu"

    def to(self, device):
        self._device = device
        return self

    def half(self):
        return self

    def eval(self):
        return self

    def forward(self, input_ids=None, attention_mask=None, **kw):
        import torch
        return FakeClassifierOutput(torch.zeros(len(input_ids), self.config.num_labels))

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class FakeTokenizer:
    """Dummy tokenizer: identity ``input_ids`` so batch size matches text count."""
    pad_token_id = 1
    eos_token_id = 2

    def __call__(self, texts, **kw):
        import torch
        return {"input_ids": torch.zeros(len(texts), 4, dtype=torch.long),
                "attention_mask": torch.ones(len(texts), 4, dtype=torch.long)}
