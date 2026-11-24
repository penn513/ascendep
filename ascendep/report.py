import json
from collections import Counter
from itertools import product
from pathlib import Path

from .registry import BACKENDS, capabilities

EP_SIZES = (8, 16, 32, 64, 128, 256)
TARGETS = (0., .2, 1., 10., 20.)
STRATEGIES = ("off", "moonep", "ultraep")


def empty_metrics():
    return {key: {"p50": None, "p95": None} for key in ("fwd_ms", "bwd_ms", "fwd_bwd_ms", "train_step_ms", "refresh_ms")}


def enumerate_matrix():
    rows = []
    for backend, ep, target, strategy in product([b for b in BACKENDS if b.device == "npu"], EP_SIZES, TARGETS, STRATEGIES):
        base = f"{backend.backend_id}/ep{ep}/maxvio{target:g}"
        rows.append({"id": f"{base}/{strategy}", "backend_id": backend.backend_id,
                     "ep": ep, "target_maxvio": target, "actual_maxvio": None,
                     "route": "balanced" if target == 0 else "skewed",
                     "strategy": strategy, "load_balancing": strategy != "off",
                     "baseline_id": f"{base}/off", "status": "not_run", "reason": backend.reason,
                     "model": {"experts": 256, "top_k": 8, "hidden": 16, "ffn": 32, "tokens_per_rank": 4096},
                     "dtype": "bfloat16", "activation_mode": "save", "topology": None,
                     "forward_backend": None, "backward_backend": None, "kernel_family": backend.kernel_family,
                     "repo": backend.repo, "commit": backend.commit, "pr_head_sha": backend.pr_head_sha,
                     "trace_hash": None, "correctness": None, "redundant_budget": 0 if strategy == "off" else None,
                     "metrics": empty_metrics(), "source_tokens_per_second": None,
                     "peak_hbm_bytes": None, "hbm_reason": "No NPU measurement",
                     "speedup": None, "peak_hbm_delta_bytes": None})
    return rows


def write_report(rows, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for row in rows:
        if row.get("strategy") != "fixed_replica_fixture" or row.get("status") != "pass":
            continue
        baseline = next((r for r in rows if r.get("strategy") == "off" and r.get("status") == "pass"
                         and all(r.get(k) == row.get(k) for k in ("backend_id", "ep", "trace_hash", "model", "dtype"))), None)
        if baseline:
            row["baseline_id"] = baseline["id"]
            row["speedup"] = {key: baseline["metrics"][key]["p50"] / row["metrics"][key]["p50"]
                              for key in ("fwd_ms", "bwd_ms", "fwd_bwd_ms", "train_step_ms")}
    report = {"schema_version": 1, "coverage": dict(Counter(row["status"] for row in rows)),
              "completed": sum(row["status"] == "pass" for row in rows), "planned": len(rows),
              "capabilities": capabilities(), "results": rows}
    (directory / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    lines = ["# ascendep results", "", f"Completed {report['completed']}/{report['planned']}. CPU results do not validate NPU support.", "",
             "Pending variant entries are inventory placeholders; native variant audit remains incomplete.", "",
             "| Backend | EP | Target | Strategy | Status | Fwd P50 ms | Bwd P50 ms | Fwd+Bwd P50 ms | Reason |",
             "|---|---:|---:|---|---|---:|---:|---:|---|"]
    for row in rows:
        times = [row["metrics"][key]["p50"] for key in ("fwd_ms", "bwd_ms", "fwd_bwd_ms")]
        cells = [row["backend_id"], row["ep"], row["target_maxvio"], row["strategy"], row["status"], *times, row.get("reason", "")]
        lines.append("| " + " | ".join("—" if cell is None else str(cell).replace("|", "\\|").replace("\n", " ") for cell in cells) + " |")
    (directory / "results.md").write_text("\n".join(lines) + "\n")
    return report
