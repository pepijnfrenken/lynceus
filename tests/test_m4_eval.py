"""Smoke test for the M4 OOD eval harness (Task C)."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmarks import _common as C
from _fakes import fake_specs, synthetic_corpus


def _write_m4(tmp_path, rows):
    p = tmp_path / "m4.jsonl"
    p.write_text("\n".join(json.dumps({"text": t, "label": l}) for t, l in rows)
                 + "\n")
    return str(tmp_path)


def test_load_m4_text_label_rows(tmp_path):
    rows = [("Human prose, no markers here", 0),
            ("In conclusion, a robust machine output!", 1)]
    d = _write_m4(tmp_path, rows)
    samples = C.load_labeled_texts(d, limit=None)
    assert len(samples) == 2
    assert [s["label"] for s in samples] == [0.0, 1.0]


def test_run_harness_on_m4_synthetic(tmp_path):
    texts, labels = synthetic_corpus(n_human=2, n_ai=2)
    d = _write_m4(tmp_path, list(zip(texts, labels)))
    samples = C.load_labeled_texts(d, limit=4)
    specs = fake_specs()
    summary = C.run_harness("m4", [s["text"] for s in samples],
                            [s["label"] for s in samples], specs, batch_size=4)
    assert summary["meta"]["n"] == 4
    assert set(summary["layers"]["ensemble"]) >= {"auroc", "macro_f1"}


def test_m4_main_data_absent_returns_2():
    import benchmarks.m4_eval as m
    rc = m.main(["--data-dir", "/nonexistent/m4", "--limit", "1"])
    assert rc == 2
