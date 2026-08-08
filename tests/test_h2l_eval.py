"""Smoke test for the H2L (human-polished-by-LLM) eval harness (Task D)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmarks import _common as C
from _fakes import fake_specs


def test_surrogate_polish_changes_text_and_is_deterministic():
    import benchmarks.h2l_eval as h
    src = h.default_human_samples()[0]
    p1 = h.surrogate_polish(src, seed=42)
    p2 = h.surrogate_polish(src, seed=42)
    assert p1 != src          # a polish was applied
    assert p1 == p2           # deterministic given the seed


def test_build_surrogate_pairs_paired_labels():
    import benchmarks.h2l_eval as h
    pairs = h.build_surrogate_pairs(None, limit=4, use_ollama=False,
                                    force_deterministic=True)
    assert pairs["note"] == "surrogate H2L (synthesized via deterministic), NOT ARB data."
    assert pairs["mechanism"] == "deterministic"
    assert len(pairs["texts"]) == 8          # 4 human + 4 polished
    assert [int(x) for x in pairs["labels"]] == [0, 1, 0, 1, 0, 1, 0, 1]


def test_run_harness_on_surrogate_h2l_4plus4():
    import benchmarks.h2l_eval as h
    pairs = h.build_surrogate_pairs(None, limit=4, use_ollama=False,
                                    force_deterministic=True)
    summary = C.run_harness("h2l-surrogate", pairs["texts"], pairs["labels"],
                            fake_specs(), batch_size=8)
    assert summary["meta"]["n"] == 8
    assert summary["meta"]["n_human"] == 4
    assert summary["meta"]["n_ai"] == 4
    assert set(summary["layers"]["ensemble"]) >= {"auroc", "macro_f1"}


def test_ollama_probe_never_crashes():
    import benchmarks.h2l_eval as h
    # reachable or not, the probe must simply return a bool
    assert isinstance(h.ollama_reachable(timeout=0.01), bool)
