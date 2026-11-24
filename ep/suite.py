"""Run sequential EP2 or EP8 jobs in a new result directory inside the project container."""
import argparse
import json
import os
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    inventory = json.loads((root / "backends.json").read_text())
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--backend", choices=inventory)
    group.add_argument("--suite", choices=["ep8_balanced", "ep2_balanced"])
    parser.add_argument("--output", type=Path, default=Path("results") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--profile", action="store_true", help="collect three steady-state forward/backward traces per rank after timing")
    args = parser.parse_args()
    world = len(os.environ["ASCEND_RT_VISIBLE_DEVICES"].split(","))
    if world not in (2, 8) or args.suite and args.suite != f"ep{world}_balanced":
        parser.error("suite must match visible NPU count")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    for name in ([args.backend] if args.backend else ["alltoall_hccl", *(key for key in inventory if key != "alltoall_hccl")]):
        meta = inventory[name]
        row = dict(backend=name, status="unsupported" if meta["mode"] == "unsupported" else "not_run", reason=meta["scope"], metrics=None)
        if meta["mode"] == "layer":
            cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={world}",
                   str(root/name/"demo.py"), "--output", str(out)]
            if args.profile:
                cmd.append("--profile")
            with (out/f"{name}.log").open("w") as log:
                env = os.environ.copy()
                if name == "mc2_dispatch_cann":
                    env.setdefault("HCCL_BUFFSIZE", "512")
                if name.endswith("_lueelu") and "DEEPEP_SHMEM_ENDPOINT" not in env:
                    with socket.socket() as listener:
                        listener.bind(("127.0.0.1", 0))
                        env["DEEPEP_SHMEM_ENDPOINT"] = f"tcp://127.0.0.1:{listener.getsockname()[1]}"
                luenv = root.parent / "deps/luenv"
                if name.endswith("_lueelu") and (luenv / "bin/python").is_file():
                    cmd[0] = str(luenv / "bin/python")
                    shmem = root.parent / "deps/lueelu/third-party/shmem/install/shmem/lib"
                    env["LD_LIBRARY_PATH"] = str(shmem) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
                epgmm = root.parent / "deps/epgmm-site"
                if name == "ep_gmm_fused_lueelu" and epgmm.is_dir():
                    env["PYTHONPATH"] = str(epgmm) + os.pathsep + env.get("PYTHONPATH", "")
                if name == "megamoe_td":
                    td = root.parent / "deps/td"
                    compiler_bin = root.parent / "deps/td-bishengir/bin"
                    if (compiler_bin / "bishengir-compile").is_file():
                        env["PATH"] = str(compiler_bin) + os.pathsep + env["PATH"]
                    cmd[0] = str(root.parent / "deps/tdenv/bin/python")
                    env["PYTHONPATH"] = os.pathsep.join([str(td / "src"), str(td / "tests"), env.get("PYTHONPATH", "")])
                sgl = root.parent / "deps/sglenv/bin/python"
                if name == "deepep_sglang" and sgl.is_file():
                    cmd[0] = str(sgl)
                    env.setdefault("HCCL_BUFFSIZE", "2048")
                    env.update(DEEP_USE_MODE="default", DEEP_NORMAL_MODE_USE_INT8_QUANT="0",
                               MOE_EXPERT_TOKEN_NUMS_TYPE="1")
                    site = root.parent / "deps/sgl-site"
                    env["PYTHONPATH"] = os.pathsep.join([str(site), str(site / "deep_ep"), env.get("PYTHONPATH", "")])
                    vendor = site / "deep_ep/vendors/hwcomputing"
                    env["LD_LIBRARY_PATH"] = str(vendor / "op_api/lib") + os.pathsep + env.get("LD_LIBRARY_PATH", "")
                moon = root.parent / "deps/moon"
                if name == "moonep" and moon.is_dir():
                    cmd[0] = str(root.parent / "deps/moonenv/bin/python")
                    env["PYTHONPATH"] = str(moon) + os.pathsep + env.get("PYTHONPATH", "")
                    sdk = moon / "npu_patch/3rdparty/shmem/install/shmem/lib"
                    env["LD_LIBRARY_PATH"] = str(sdk) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
                fsdp = root.parent / "deps/fsdp"
                if name == "megamoe_fsdpturbo" and (root.parent / "deps/fsdp-mega").is_dir():
                    fsdp = root.parent / "deps/fsdp-mega"
                    fm = root.parent / "deps/fmenv/bin/python"
                    if fm.is_file():
                        cmd[0] = str(fm)
                        fm_source = root.parent / "deps/fsdp-train"
                        env["PYTHONPATH"] = str(fm_source) + os.pathsep + env.get("PYTHONPATH", "")
                        env["LD_LIBRARY_PATH"] = str(fm_source / "ascend_deepep/lib") + os.pathsep + env.get("LD_LIBRARY_PATH", "")
                        env.setdefault("ASCEND_DEEPEP_TOKEN_CHUNK_M", "4096")
                if name.endswith("_fsdpturbo") and fsdp.is_dir():
                    env["PYTHONPATH"] = str(fsdp) + os.pathsep + env.get("PYTHONPATH", "")
                mcore = root.parent / "deps/mcore"
                if name == "allgather_overlap_mindspeed":
                    env["PYTHONPATH"] = os.pathsep.join([str(root.parent / "deps/minds"), str(mcore), env.get("PYTHONPATH", "")])
                if name.endswith("_megatron") and mcore.is_dir():
                    env["PYTHONPATH"] = str(mcore) + os.pathsep + env.get("PYTHONPATH", "")
                rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
            path = out/f"{name}.json"
            if rc == 0 and path.is_file():
                row.update(json.loads(path.read_text()), reason="")
            else:
                row.update(status="failed", reason=f"exit {rc}, see {name}.log")
        rows.append(row)
        (out/"results.json").write_text(json.dumps(rows, indent=2))
        lines = ["| Backend | Status | Fwd P50 ms | Bwd P50 ms | Fwd+Bwd P50 ms | Tokens/s | Peak allocated bytes | Reason |",
                 "|---|---|---:|---:|---:|---:|---:|---|"]
        for r in rows:
            m = r["metrics"] or {}
            times = [m.get(k, {}).get("p50", "—") for k in ("fwd_ms","bwd_ms","fwd_bwd_ms")]
            peak = max((v["allocated"] for v in r.get("ranks", [])),default="—")
            lines.append("| " + " | ".join(map(str,[r["backend"],r["status"],*times,
                         r.get("source_tokens_per_second","—"),peak,r["reason"]])) + " |")
        (out/"results.md").write_text("\n".join(lines)+"\n")
    if args.backend and rows[0]["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
