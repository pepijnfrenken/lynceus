"""Desklib detector — desklib/ai-text-detector-v1.01 (434M, DeBERTa-v3-large)."""

import torch
from .base import Detector


class DesklibDetector(Detector):
    """DeBERTa-v3-large with mean-pooling + linear head. #1 on RAID leaderboard."""

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._device = None
        self._model_id = "desklib/ai-text-detector-v1.01"

    def load(self, device: str, fp16: bool = False) -> None:
        import torch.nn as nn
        from transformers import AutoTokenizer, AutoConfig, AutoModel
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        class DesklibHead(nn.Module):
            def __init__(self, base):
                super().__init__()
                self.base = base
                self.classifier = nn.Linear(base.config.hidden_size, 1)

            def forward(self, input_ids, attention_mask=None, **kw):
                h = self.base(input_ids=input_ids, attention_mask=attention_mask)[0]
                mask = attention_mask.unsqueeze(-1).expand(h.size()).float()
                pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                pooled = pooled.to(self.classifier.weight.dtype)
                return {"logits": self.classifier(pooled)}

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_id)
        state = load_file(hf_hub_download(self._model_id, "model.safetensors"))
        base_state = {k.removeprefix("model."): v for k, v in state.items() if not k.startswith("classifier")}
        config = AutoConfig.from_pretrained(self._model_id)
        base = AutoModel.from_config(config)
        base.load_state_dict(base_state, strict=True)
        model = DesklibHead(base)
        cls_w = {k.removeprefix("classifier."): v for k, v in state.items() if k.startswith("classifier.")}
        if cls_w:
            model.classifier.load_state_dict(cls_w)
        if fp16:
            model = model.half()
        self._device = device
        self._model = model.to(device).eval()

    def score(self, sentences: list[str], batch_size: int = 32) -> list[float]:
        probs = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]
            enc = self._tokenizer(batch, padding=True, truncation=True, max_length=768, return_tensors="pt")
            with torch.no_grad():
                logits = self._model(
                    input_ids=enc["input_ids"].to(self._device),
                    attention_mask=enc["attention_mask"].to(self._device),
                )["logits"]
            batch_probs = torch.sigmoid(logits).squeeze(-1).cpu().tolist()
            if isinstance(batch_probs, float):
                batch_probs = [batch_probs]
            probs.extend(batch_probs)
        return probs

    @property
    def name(self) -> str:
        return "desklib"

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def description(self) -> str:
        return "Desklib 434M — #1 on RAID leaderboard, most robust vs adversarial"

    def set_model_id(self, model_id: str) -> None:
        self._model_id = model_id
