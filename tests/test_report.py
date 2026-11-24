import json

from ascendep.benchmark import benchmark
from ascendep.model import MoEConfig
from ascendep.registry import BACKENDS
from ascendep.report import EP_SIZES, STRATEGIES, TARGETS, enumerate_matrix, write_report
from ascendep.routing import make_trace


def test_matrix_coverage_and_nulls(tmp_path):
    rows = enumerate_matrix()
    backends = [b for b in BACKENDS if b.device == "npu"]
    assert len(rows) == len(backends) * 90 == 990
    assert len({row["id"] for row in rows}) == len(rows)
    for backend in backends:
        actual = {(r["ep"], r["target_maxvio"], r["strategy"]) for r in rows if r["backend_id"] == backend.backend_id}
        expected = {(e, t, s) for e in EP_SIZES for t in TARGETS for s in STRATEGIES}
        assert actual == expected
    assert all(row["status"] == "not_run" and row["reason"] for row in rows)
    assert all(row["metrics"][key]["p50"] is None for row in rows for key in ("fwd_ms", "bwd_ms", "fwd_bwd_ms"))
    write_report(rows, tmp_path)
    report = json.loads((tmp_path / "results.json").read_text())
    assert report["completed"] == 0 and report["planned"] == 990
    assert "not_run" in (tmp_path / "results.md").read_text()


def test_benchmark_independent_total_and_raw_records(tmp_path):
    config = MoEConfig(hidden=4, ffn=7)
    trace = make_trace(config)
    row = benchmark(config, trace, warmup=1, samples=3)
    assert len(row["raw"]) == 3
    for key in ("fwd_ms", "bwd_ms", "fwd_bwd_ms"):
        assert row["metrics"][key]["p50"] > 0
    assert row["source_tokens_per_second"] > 0
    assert row["peak_hbm_bytes"] is None
    write_report([row], tmp_path)
