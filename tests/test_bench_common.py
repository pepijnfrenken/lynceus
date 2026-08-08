"""Unit tests for the shared benchmark metrics module."""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmarks import _common as C
from _fakes import FakeDetector, synthetic_corpus


def test_auroc_perfect_separation():
    assert C.auroc([0.1, 0.9, 0.8], [0, 1, 1]) == 1.0


def test_auroc_tied_all_equal_means_random():
    assert C.auroc([0.5, 0.5, 0.5, 0.5], [0, 0, 1, 1]) == 0.5


def test_auroc_ties_across_classes_records():
    # Ties straddle both classes -> no separating signal -> AUROC 0.5
    val = C.auroc([0.8, 0.8, 0.1, 0.1], [1, 0, 1, 0])
    assert abs(val - 0.5) < 1e-9


def test_auroc_all_pos_above_neg_with_self_ties():
    assert C.auroc([0.1, 0.9, 0.8, 0.8, 0.2], [0, 1, 1, 1, 0]) == 1.0


def test_tpr_at_fpr():
    scores = [0.1, 0.95, 0.9, 0.2, 0.85, 0.3]
    labels = [0, 1, 1, 0, 1, 0]
    assert C.tpr_at_fpr(scores, labels, 0.05) == 1.0


def test_macro_f1_perfect():
    assert C.macro_f1([0.9, 0.95, 0.2, 0.3], [1, 1, 0, 0]) == 1.0


def test_macro_f1_all_tied_finite():
    m = C.macro_f1([0.5, 0.5, 0.5, 0.5], [0, 0, 1, 1])
    assert 0.0 <= m <= 1.0


def test_sample_metrics_keys():
    m = C.sample_metrics([0.1, 0.9, 0.8, 0.2], [0, 1, 1, 0])
    assert set(m) == {"auroc", "tpr_at_5_%_fpr", "macro_f1"}


def test_compute_layer_metrics_structure():
    texts, labels = synthetic_corpus()  # default 4 human + 4 AI
    det_results = C.run_detectors(
        [("detA", FakeDetector("A"))], texts, batch_size=8
    )
    layers = C.compute_layer_metrics(texts, labels, det_results)
    assert set(layers) >= {"lexical", "burstiness", "classifier", "ensemble"}
    assert set(layers["lexical"]) >= {"auroc", "precision", "recall", "f1"}
    assert set(layers["burstiness"]) >= {"auroc"}
    assert "detA" in layers["classifier"]
    assert set(layers["ensemble"]) >= {"auroc", "macro_f1"}


def test_check_data_present_raises():
    with pytest.raises(C.DataError):
        C.check_data_present("/nonexistent/definitely/missing", "*.jsonl")


def _pair_dir(base, name):
    d = base / name
    d.mkdir()
    return d


def test_load_labeled_texts_pair_and_label_formats(tmp_path):
    raid_dir = _pair_dir(tmp_path, "raidp")
    raid = raid_dir / "raid.jsonl"
    raid.write_text(json.dumps({"id": "a", "human_text": "hello good morning",
                                "ai_text": "In conclusion, a robust result."})
                    + "\n")
    samples = C.load_labeled_texts(str(raid_dir), limit=None)
    assert len(samples) == 2
    assert samples[0]["label"] == 0.0
    assert samples[1]["label"] == 1.0

    m4_dir = _pair_dir(tmp_path, "m4p")
    m4 = m4_dir / "m4.jsonl"
    m4.write_text(json.dumps({"text": "some human prose", "label": 0}) + "\n" +
                  json.dumps({"text": "machine output here", "label": 1}) + "\n")
    labels2 = [s["label"] for s in C.load_labeled_texts(str(m4_dir), limit=None)]
    assert labels2.count(0.0) == 1
    assert labels2.count(1.0) == 1


def test_limit_applies(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text("\n".join(json.dumps({"text": "x", "label": i}) for i in [0, 1, 0, 1]))
    samples = C.load_labeled_texts(str(tmp_path), limit=2)
    assert len(samples) == 2


def test_windows_path_write_json(tmp_path):
    out = tmp_path / "out.json"
    C.write_json(str(out), {"a": 1, "b": ["x", 2]})
    assert json.loads(out.read_text())["b"] == ["x", 2]
