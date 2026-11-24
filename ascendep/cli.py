import argparse
import json
import os
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from .benchmark import benchmark
from .model import MoEConfig
from .launch import run_local
from .registry import capabilities
from .report import EP_SIZES, enumerate_matrix, write_report
from .routing import Trace, make_trace, stream_route_summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="CPU MoE validation and NPU coverage inventory")
    subs = parser.add_subparsers(dest="command", required=True)
    subs.add_parser("capabilities")
    launch = subs.add_parser("launch", help="Run a local two-process torchrun job")
    launch.add_argument("--module", action="store_true")
    launch.add_argument("arguments", nargs=argparse.REMAINDER)
    matrix = subs.add_parser("matrix")
    matrix.add_argument("--output", default="reports/matrix")
    audit = subs.add_parser("route-audit", help="Stream large routes without allocating model weights")
    audit.add_argument("--experts", type=int, default=2048)
    audit.add_argument("--top-k", type=int, default=16)
    audit.add_argument("--tokens-per-rank", type=int, default=4096)
    audit.add_argument("--targets", nargs="+", type=float, default=[0.])
    audit.add_argument("--output", default="reports/route-audit.json")
    regress = subs.add_parser("regress")
    regress.add_argument("--output", default="reports/regression")
    for command in ("trace", "benchmark"):
        cmd = subs.add_parser(command)
        cmd.add_argument("--config", help="Model JSON; defaults to the built-in CPU small shape")
        cmd.add_argument("--target", type=float, default=0.)
        cmd.add_argument("--tokens-per-rank", type=int, default=8)
        cmd.add_argument("--seed", type=int, default=31)
        cmd.add_argument("--output", default=f"reports/{command}")
        if command == "trace":
            cmd.add_argument("--world", type=int, default=2)
        else:
            cmd.add_argument("--backend", choices=("cpu/reference", "cpu/gloo"), default="cpu/reference")
            cmd.add_argument("--trace", help="Replay a trace instead of generating inputs")
            cmd.add_argument("--fixed-replica", action="store_true", help="Replicate expert 0 on rank 1; CPU semantic fixture")
            cmd.add_argument("--warmup", type=int, default=20)
            cmd.add_argument("--samples", type=int, default=50)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    if args.command == "launch":
        if not args.arguments:
            parser.error("launch requires a script or module and its arguments")
        run_local(args.arguments, module=args.module)
        return
    if args.command == "capabilities":
        print(json.dumps(capabilities(), indent=2))
        return
    if args.command == "matrix":
        result = write_report(enumerate_matrix(), args.output)
        print(json.dumps(result["coverage"]))
        return
    if args.command == "route-audit":
        rows = [stream_route_summary(args.experts, args.top_k, args.tokens_per_rank, ep, target)
                for ep in EP_SIZES for target in args.targets]
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rows, indent=2) + "\n")
        print(json.dumps([{k: row[k] for k in ("world", "target_maxvio", "actual_maxvio", "status")} for row in rows], indent=2))
        return
    if args.command == "regress":
        root = Path(__file__).resolve().parents[1]
        output = Path(args.output).resolve()
        os.environ["ASCENDEP_VALIDATION_REPORT"] = str(output / "training-validation.json")
        subprocess.run([sys.executable, "-m", "pytest", "-q", str(root / "tests")], check=True, cwd=root, timeout=240)
        base = [sys.executable, "-m", "ascendep", "benchmark", "--warmup", "2", "--samples", "5"]
        subprocess.run(base + ["--output", str(output / "reference")], check=True, timeout=120)
        run_local(["ascendep"] + base[3:] + ["--backend", "cpu/gloo", "--output", str(output / "gloo")], module=True, timeout=120)
        run_local(["ascendep"] + base[3:] + ["--backend", "cpu/gloo", "--fixed-replica", "--output", str(output / "replica")], module=True, timeout=120)
        rows = enumerate_matrix()
        for name in ("reference", "gloo", "replica"):
            rows.extend(json.loads((output / name / "results.json").read_text())["results"])
        write_report(rows, output)
        audits = [stream_route_summary(2048, 16, 4096, ep) for ep in EP_SIZES]
        (output / "large-route-audit.json").write_text(json.dumps(audits, indent=2) + "\n")
        print(f"Regression passed. Reports: {output}")
        return
    config = MoEConfig(**json.loads(Path(args.config).read_text())) if args.config else MoEConfig()
    distributed = args.command == "benchmark" and args.backend == "cpu/gloo"
    initialized_here = False
    try:
        if distributed:
            if "RANK" not in os.environ:
                parser.error("cpu/gloo must be launched with torchrun --nproc-per-node=2")
            dist.init_process_group("gloo", timeout=timedelta(seconds=60))
            initialized_here = True
            world, rank = dist.get_world_size(), dist.get_rank()
        else:
            world, rank = (args.world if args.command == "trace" else 1), 0
        trace = Trace.load(args.trace) if getattr(args, "trace", None) else make_trace(config, args.tokens_per_rank, world, args.target, args.seed)
        trace.validate(config)
        if args.command == "trace":
            trace.save(args.output)
            print(json.dumps({"hash": trace.digest(), **trace.metadata}, indent=2))
            return
        if args.fixed_replica and (not distributed or world != 2):
            parser.error("fixed replica fixture requires EP2 cpu/gloo")
        row = benchmark(config, trace, args.backend, {0: (1,)} if args.fixed_replica else None, args.warmup, args.samples)
        if rank == 0:
            trace.save(Path(args.output) / "trace.pt")
            write_report([row], args.output)
            print(json.dumps({"status": row["status"], "correctness": row["correctness"], "metrics": row["metrics"]}, indent=2))
    finally:
        if initialized_here:
            dist.destroy_process_group()
