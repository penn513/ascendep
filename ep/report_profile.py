"""Verify local profile archives and index EP8 benchmark attempts."""
import argparse
import csv
import io
from collections import Counter
from statistics import median
import hashlib
import json
import tarfile
from ep.archive_profile import inspect_events
from pathlib import Path


def profile_summary(package):
    step_rows = []
    waiting = []
    operators = Counter()
    # Sequential decompression avoids seeking backwards through large gzip streams.
    with tarfile.open(package, "r|gz") as archive:
        for member in archive:
            if not member.isfile() or "/ASCEND_PROFILER_OUTPUT/" not in member.name:
                continue
            filename = Path(member.name).name
            if filename == "trace_view.json":
                inspect_events(json.loads(archive.extractfile(member).read()), member.name)
            if filename not in {"step_trace_time.csv", "op_statistic.csv"}:
                continue
            data = archive.extractfile(member).read().decode("utf-8-sig")
            rows = list(csv.DictReader(io.StringIO(data)))
            if filename == "op_statistic.csv":
                for row in rows:
                    operators[row["OP Type"]] += float(row["Total Time(us)"])
            elif rows:
                step_rows.extend(rows)
                waiting.append(median(float(r["Free"]) for r in rows) / 1000)
    fields = ["Communication(Not Overlapped)", "Overlapped"]
    comm = [f"{median(float(r[field]) for r in step_rows)/1000:.2f}" for field in fields] if step_rows else ["—", "—"]
    free = f"{min(waiting):.2f}–{max(waiting):.2f}" if waiting else "—"
    return [", ".join(name for name, _ in operators.most_common(3)), *comm, free]


def render(directory, run=None):
    if run is None:
        rows = json.loads((directory / "results.json").read_text())
    else:
        rows = json.loads((run / "results.json").read_text())
        for index, original in enumerate(rows):
            name = original["backend"]
            retry = run / f"retry-{name}" / "results.json"
            if retry.is_file():
                attempts = json.loads(retry.read_text())
                if len(attempts) != 1 or attempts[0]["backend"] != name:
                    raise ValueError(f"invalid retry manifest for {name}")
                rows[index] = dict(attempts[0], source=str(retry.relative_to(run)))
        (directory / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    inventory = json.loads(Path(__file__).with_name("backends.json").read_text())
    if {r["backend"] for r in rows} != set(inventory) or len(rows) != len(inventory):
        raise ValueError("incomplete backend inventory")
    table, failed, checksums = [], [], []
    profile_rows = []
    hashes = None
    for row in rows:
        name = row["backend"]
        if row["status"] != "pass":
            reason = row["reason"]
            log = directory / f"{name}.log"
            if log.is_file():
                lines = log.read_text(errors="replace").splitlines()
                reason = next((line for line in lines if any(tag in line for tag in
                               ("ValueError:", "RuntimeError:", "ModuleNotFoundError:", "AssertionError:"))), reason)
            failed.append(f"- `{name}` · {reason}")
            continue
        meta = json.loads((directory / f"{name}.tar.json").read_text())
        package = directory / meta["archive"]
        with package.open("rb") as stream:
            checksum = hashlib.sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                checksum.update(block)
            digest = checksum.hexdigest()
        if digest != meta["sha256"]:
            raise ValueError(f"{name}: invalid archive checksum")
        checksums.append(f"{digest}  {package.name}")
        with tarfile.open(package) as archive:
            content = set(archive.getnames())
            benchmark = json.load(archive.extractfile(f"{name}.json"))
            if benchmark["metrics"] != row["metrics"] or benchmark["metrics"] != meta["metrics"]:
                raise ValueError(f"{name}: archive and benchmark disagree")
            for record in meta["ranks"]:
                if record["trace"] not in content or record["markers"] != {"ep_forward": 3, "ep_backward": 3} or record["device_events"] <= 0:
                    raise ValueError(f"{name}: invalid rank trace")
        ranks = sorted(benchmark["ranks"], key=lambda r: r["rank"])
        trace = [r["trace_sha256"] for r in ranks]
        if hashes is None:
            hashes = trace
        if (benchmark["shape"] != [4096, 7168, 3072] or benchmark["local_experts"] != 3
                or benchmark["ep"] != 8 or benchmark["top_k"] != 6 or benchmark["experts"] != 24
                or benchmark["dtype"] != "bfloat16" or benchmark["planner"] != "off"
                or benchmark["warmup"] != 20 or benchmark["samples"] != 50 or trace != hashes
                or benchmark["physical_device_nodes"] != [f"davinci{i}" for i in range(8)]
                or [r["rank"] for r in ranks] != list(range(8))
                or [r["rank"] for r in meta["ranks"]] != list(range(8))
                or [r["trace_sha256"] for r in meta["ranks"]] != trace):
            raise ValueError(f"{name}: inconsistent EP8 configuration or input trace")
        for rank in ranks:
            if len(rank["raw_ms"]) != 50 or any(len(v) != 3 or min(v) <= 0 for v in rank["raw_ms"]):
                raise ValueError(f"{name}: incomplete timings")
            for stage in ("cpu_fp32_small", "hccl_target_shape"):
                errors = rank["validation"][stage]
                if len(errors) != 5 or not all(0 <= x < .035 for x in errors):
                    raise ValueError(f"{name}: failed numerical check")
        profile_rows.append("| " + " | ".join([name, *profile_summary(package)]) + " |")
        m = benchmark["metrics"]
        times = [f"{m[k]['p50']:.2f} / {m[k]['p95']:.2f}" for k in ("fwd_ms", "bwd_ms", "fwd_bwd_ms")]
        table.append((m["fwd_bwd_ms"]["p50"], f"| {name} | " + " | ".join(times) +
                      f" | {benchmark['source_tokens_per_second']:.0f} | [{package.name}]({package.name}) |"))
    lines = ["# Ascend 950DT · EP8 benchmark 与 profiling", "",
             f"{len(table)} 个后端完成性能采样并归档八个 rank 的 profiling。共 {len(inventory)} 个候选。",
             "", "物理卡 0–7，BF16，均衡路由，planner 关闭。H7168、FFN3072、Top-6、24 个专家，每卡 3 个专家、4096 tokens。吞吐按全组 32768 个源 tokens 计算。",
             "", "预热 20 轮、采样 50 轮。每轮取八个 rank 的最大耗时，再计算 P50/P95。性能采样结束后开启 Level1 CPU/NPU profiling，预热 1 步，记录 3 步。",
             "", "| 后端 | 前向 P50/P95 ms | 反向 P50/P95 ms | 前反向 P50/P95 ms | tokens/s | Profiling 归档 |",
             "|---|---:|---:|---:|---:|---|", *(row for _, row in sorted(table)),
             "", "每个压缩包包含 benchmark JSON、运行日志、各 rank 的原始 profiling 及解析结果。解压后在 MindStudio Insight 中打开对应 rank 的 trace_view.json，搜索 ep_forward、ep_backward 可定位主机阶段，并沿关联事件查看 NPU 执行。",
             "", "## Profiling 观察", "",
             "主要算子按 profiler 的累计设备执行时间排序。通信与重叠取 step_trace_time 中各 rank 各步的中位数。设备空闲列为各 rank 三步空闲时长中位数的范围。SHMEM 等融合算子内部通信可能计入算子耗时；可见通信为零时仍可能存在内部传输。",
             "", "| 后端 | 累计耗时前三算子类型 | 可见未重叠通信 ms | 可见重叠 ms | 设备空闲范围 ms |",
             "|---|---|---:|---:|---:|", *profile_rows,
             "", "## 未取得完整结果", "", *failed]
    (directory / "benchmark.md").write_text("\n".join(lines) + "\n")
    (directory / "SHA256SUMS").write_text("\n".join(checksums) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--run", type=Path, help="suite output with optional retry-<backend> subdirectories")
    args = parser.parse_args()
    render(args.directory, args.run)
