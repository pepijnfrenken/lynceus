"""Smoke test for scripts/bench_baseline.py (Task A)."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scripts.bench_baseline as bench
from _fakes import fake_specs


def test_build_sample_size():
    s = bench.build_sample(20)
    assert len(s) == 40
    assert len(bench.build_sample(2)) == 4


def test_run_baseline_structure():
    specs = fake_specs()  # tmr, desklib, fast-detect, binoculars(degen)
    out = bench.run_baseline(specs, n_each=2, batch_size=4)
    assert out["n_human"] == 2
    assert out["n_ai"] == 2
    assert set(out["per_detector"]) == {"tmr", "desklib", "fast-detect",
                                        "binoculars(degen)"}
    for name, d in out["per_detector"].items():
        assert set(d) >= {"mean", "std", "min", "max"}
    assert set(out["ensemble"]) >= {"mean", "std"}


def test_run_baseline_json_roundtrip(tmp_path):
    out_json = str(tmp_path / "baseline.json")
    summary = bench.run_baseline(fake_specs(), n_each=2, batch_size=4)
    import benchmarks._common as C
    C.write_json(out_json, summary)
    loaded = json.loads(open(out_json).read())
    assert loaded["ensemble"]["mean"] == summary["ensemble"]["mean"]
