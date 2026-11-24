"""Validate and archive one backend's EP benchmark and per-rank Ascend traces."""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path


def inspect_trace(path):
    data = json.loads(path.read_bytes())
    return inspect_events(data, path)


def inspect_events(data, path):
    events = data if isinstance(data, list) else data["traceEvents"]
    markers = {key: 0 for key in ("ep_forward", "ep_backward")}
    devices = set()
    for event in events:
        if event.get("ph") == "M" and event.get("name") == "process_name":
            label = str(event.get("args", {}).get("name", "")).lower()
            if "ascend hardware" in label:
                devices.add(event["pid"])
        name = event.get("name")
        if name in markers and event.get("ph") == "X":
            markers[name] += 1
    kernels = sum(1 for event in events if event.get("pid") in devices and
                  event.get("ph") == "X" and float(event.get("dur", 0)) > 0)
    if list(markers.values()) != [3, 3] or not kernels:
        raise ValueError(f"{path}: markers={markers}, device_events={kernels}")
    steps = [e for e in events if e.get("ph") == "X" and e.get("name", "").startswith("ProfilerStep#")]
    if len(steps) != 3:
        raise ValueError(f"{path}: expected three active steps, found {len(steps)}")
    for step in steps:
        start = float(step["ts"])
        end = start + float(step["dur"])
        ordered = sorted((float(e["ts"]), e["name"]) for e in events if e.get("ph") == "X"
                         and e.get("name") in markers and start <= float(e["ts"]) < end)
        device_active = any(e.get("pid") in devices and e.get("ph") == "X" and float(e.get("dur", 0)) > 0
                            and start <= float(e["ts"]) < end for e in events)
        if [name for _, name in ordered] != ["ep_forward", "ep_backward"] or not device_active:
            raise ValueError(f"{path}: incomplete {step['name']}")
    return dict(markers=markers, device_events=kernels)


def archive(directory, backend, destination):
    benchmark_path = directory / f"{backend}.json"
    result = json.loads(benchmark_path.read_bytes())
    world = result["ep"]
    ranks = sorted(result["ranks"], key=lambda r: r["rank"])
    if result["status"] != "pass" or result["backend"] != backend or [r["rank"] for r in ranks] != list(range(world)):
        raise ValueError("incomplete benchmark")
    profiles = directory / "profiling" / backend
    manifest = []
    for record in ranks:
        rank = record["rank"]
        traces = list((profiles / f"rank{rank}").rglob("trace_view.json"))
        if len(traces) != 1:
            raise ValueError(f"rank {rank}: expected one parsed timeline, found {len(traces)}")
        manifest.append(dict(rank=rank, trace=str(traces[0].relative_to(directory)),
                             trace_sha256=record["trace_sha256"], **inspect_trace(traces[0])))
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / f"{backend}.tar.gz"
    partial = target.with_suffix(".partial")
    with tarfile.open(partial, "w:gz", compresslevel=1) as bundle:
        for path in [benchmark_path, directory / f"{backend}.log", profiles]:
            bundle.add(path, arcname=str(path.relative_to(directory)))
    digest = hashlib.sha256()
    with partial.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    partial.replace(target)
    index = dict(backend=backend, ep=world, archive=target.name, bytes=target.stat().st_size,
                 sha256=digest.hexdigest(), metrics=result["metrics"], ranks=manifest)
    target.with_suffix(".json").write_text(json.dumps(index, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("backend")
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    archive(args.directory, args.backend, args.destination)
