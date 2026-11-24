"""Summarize verified two-rank records and retain unmeasured entries."""
import argparse
import json
from pathlib import Path


def render(directory):
    inventory = json.loads((Path(__file__).with_name("backends.json")).read_text())
    rows = []
    failures = []
    summary = []
    for name, meta in inventory.items():
        retries = [directory / f"retry-{name}"]
        numbered = directory.glob(f"retry-[0-9]*-{name}")
        retries.extend(sorted(numbered, key=lambda p: int(p.name.split("-", 2)[1])))
        attempts = [directory, *retries]
        runs = []
        for attempt in attempts:
            manifest = attempt / "results.json"
            if manifest.is_file():
                for item in json.loads(manifest.read_text()):
                    if item["backend"] == name:
                        runs.append((item["status"], attempt))
        status, where = runs[-1] if runs else ("not_run", directory)
        path = where / (name + ".json")
        if status != "pass" or not path.exists():
            log = where / (name + ".log")
            reason = meta["scope"] if status in ("not_run", "unsupported") else status
            if log.exists():
                lines = log.read_text(errors="replace").splitlines()
                reason = next((l for l in lines if any(t in l for t in (
                    "ModuleNotFoundError:", "ImportError:", "RuntimeError:", "ValueError:", "AssertionError:", "AttributeError:"))), "详见原始日志")
            summary.append(dict(backend=name, status=status, reason=reason, metrics=None))
            failures.append((name, reason))
            continue
        r = json.loads(path.read_text())
        assert r["backend"] == name and r["status"] == "pass" and r["ep"] == 2
        assert r["shape"] == [4096, 7168, 3072] and r["experts"] == 24 and r["top_k"] == 6
        assert r["physical_device_nodes"] == ["davinci6", "davinci7"]
        assert r["visible_devices"] in ("0,1", "6,7") and r["warmup"] == 20 and r["samples"] == 50
        assert sorted(v["rank"] for v in r["ranks"]) == [0, 1]
        for rank in r["ranks"]:
            assert len(rank["raw_ms"]) == 50
            assert all(len(values) == 3 and min(values) > 0 for values in rank["raw_ms"])
            for stage in ("cpu_fp32_small", "hccl_target_shape"):
                errors = rank["validation"][stage]
                assert len(errors) == 5 and all(0 <= e < .035 for e in errors)
        summary.append(dict(backend=name, status="pass", source=str(path.relative_to(directory)),
                            metrics=r["metrics"], source_tokens_per_second=r["source_tokens_per_second"]))
        rows.append(r)
    rows.sort(key=lambda r: r["metrics"]["fwd_bwd_ms"]["p50"])
    if rows:
        assert all([v["trace_sha256"] for v in r["ranks"]] == [v["trace_sha256"] for v in rows[0]["ranks"]] for r in rows)
    lines = ["# EP2 · Ascend 950DT · 物理卡 6、7", "",
             "测试在共享卡上进行。期间采样观察到 Ray 和 vLLM 进程，时延与吞吐反映本轮环境。", "",
             "BF16，均衡路由，planner 关闭，H7168 / FFN3072，E24 / Top-6，每卡 4096 tokens、12 个专家。吞吐按全组 8192 个源 tokens 计算。", "",
             "| 后端 | 前向 P50/P95 ms | 反向 P50/P95 ms | 前反向 P50/P95 ms | tokens/s | 峰值 allocated / reserved GiB |",
             "|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        times = [f"{r['metrics'][k]['p50']:.2f} / {r['metrics'][k]['p95']:.2f}" for k in ("fwd_ms","bwd_ms","fwd_bwd_ms")]
        mem = [max(v[k] for v in r["ranks"])/2**30 for k in ("allocated","reserved")]
        lines.append("| " + " | ".join([r["backend"], *times, f"{r['source_tokens_per_second']:.0f}",f"{mem[0]:.2f} / {mem[1]:.2f}"])+" |")
    if any(r["backend"] == "megamoe_td" for r in rows):
        lines += ["", "megamoe_td 使用原生 SwiGLU。计时前对固定输入的全部路由检查限幅条件，确认小形状和目标形状均未触发阈值 10。"]
    if any(r["backend"] == "mc2_dispatch_cann" for r in rows):
        lines += ["", "mc2_dispatch_cann 每卡 4096 tokens 按 8 × 512 分块。当前 DispatchV2 单次 BS 上限为 512，计时包含八块通信、专家计算、路由权重及反向。"]
    lines += ["", "每个后端预热 20 轮、采样 50 轮，每轮取两个 rank 中较慢的耗时。前反向联合耗时独立采样。峰值显存是本进程 PyTorch 分配器在两卡上的较大值。小规模与 FP32 CPU 参考比较，目标形状与 HCCL AllToAll 比较，均校验输出、输入梯度、路由权重梯度和两组专家权重梯度。", "", "## 未取得同口径数值", ""]
    lines += [f"- {n}：{reason}" for n, reason in failures]
    (directory/"benchmark.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    (directory/"benchmark.md").write_text("\n".join(lines)+"\n")
    return len(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    print(render(parser.parse_args().directory), "validated results")
