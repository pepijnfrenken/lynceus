"""Smoke test for the RAID eval harness (Task B)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmarks import _common as C
from _fakes import fake_specs, synthetic_corpus


def test_run_harness_on_synthetic_pair():
    texts, labels = synthetic_corpus(n_human=2, n_ai=2)
    specs = fake_specs()
    summary = C.run_harness("raid", texts, labels, specs, batch_size=4,note="x")
    assert summary["meta"]["n"] == 4
    assert summary["meta"]["n_human"] == 2
    assert summary["meta"]["n_ai"] == 2
    assert "binoculars(degen)" in summary["detectors"]
    assert set(summary["layers"]["ensemble"]) >= {"auroc"}


def test_main_data_absent_returns_2():
    import benchmarks.raid_eval as r
    rc = r.main(["--data-dir", "/nonexistent/raid", "--limit", "1"])
    assert rc == 2


def test_handle_fetch_warns_and_never_downloads(tmp_path, capsys):
    import benchmarks.raid_eval as r
    parser = C.add_eval_args("prog", "desc", "benchmarks/data/raid", "raid")
    args = parser.parse_args(["--fetch", "--data-dir", str(tmp_path)])
    C.handle_fetch(args, "raid", "https://huggingface.co/datasets/liamdugan/raid")
    out = capsys.readouterr().out
    assert "fetch" in out.lower() and "manual" in out.lower()
    # nothing physically downloaded into tmp_path
    assert list(tmp_path.iterdir()) == []
