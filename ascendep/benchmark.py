"""CPU wall-clock regression timings, not accelerator performance claims."""
import platform
import resource
import time
from dataclasses import asdict

import torch
import torch.distributed as dist

from .distributed import ExpertParallelMoE
from .model import ReferenceMoE
from .report import empty_metrics
from .validation import validate_training


def benchmark(config, trace, backend="cpu/reference", replicas=None, warmup=2, samples=5):
    if samples < 1 or warmup < 0:
        raise ValueError("samples must be positive and warmup nonnegative")
    distributed = backend == "cpu/gloo"
    if backend not in ("cpu/reference", "cpu/gloo"):
        raise ValueError("backend is not implemented")
    rank = dist.get_rank() if distributed else 0
    world = dist.get_world_size() if distributed else 1
    if trace.metadata["world"] != world:
        raise ValueError("trace world does not match execution")
    trace.validate(config)
    if trace.metadata["routing"]["status"] not in ("matched", "discrete_limit"):
        raise ValueError("routing target not attained: " + trace.metadata["routing"]["status"])
    if distributed:
        correctness = validate_training(config, [trace, trace], replicas)
        model = ExpertParallelMoE(config, replicas=replicas)
        owned = list(model.owned_parameters())
    else:
        model = ReferenceMoE(config)
        owned = list(model.parameters())
        # Independently checked by unit tests, rather than comparing a layer to itself.
        correctness = {"status": "reference", "reason": "Independent numerical oracle; see CPU regression tests."}
    optimizer = torch.optim.SGD(owned, lr=.001)
    mask = trace.source_ranks == rank
    x, weights = trace.x[mask].clone().requires_grad_(), trace.weights[mask].clone().requires_grad_()
    ids = trace.ids[mask]
    grad = torch.linspace(-.7, .9, trace.x.numel(), dtype=trace.x.dtype).reshape_as(trace.x)[mask] / trace.x.numel()
    raw = []

    def prepare():
        model.zero_grad(set_to_none=True)
        x.grad, weights.grad = None, None
        if distributed:
            dist.barrier()

    def backward(out):
        out.backward(grad)
        if distributed:
            model.return_replica_gradients()

    for iteration in range(warmup + samples):
        prepare()
        start = time.perf_counter_ns()
        out = model(x, ids, weights)
        forward_end = time.perf_counter_ns()
        backward(out)
        end = time.perf_counter_ns()
        forward_ms, backward_ms = (forward_end - start) / 1e6, (end - forward_end) / 1e6
        # Separate whole-iteration measurement; never add percentile summaries.
        prepare()
        start = time.perf_counter_ns()
        backward(model(x, ids, weights))
        total_ms = (time.perf_counter_ns() - start) / 1e6
        prepare()
        start = time.perf_counter_ns()
        out = model(x, ids, weights)
        loss = (out * grad).sum()
        loss.backward()
        if distributed:
            model.return_replica_gradients()
        optimizer.step()
        refresh_start = time.perf_counter_ns()
        if distributed:
            model.refresh_replicas()
        step_end = time.perf_counter_ns()
        if iteration >= warmup:
            raw.append({"iteration": iteration - warmup, "rank": rank, "fwd_ms": forward_ms,
                        "bwd_ms": backward_ms, "fwd_bwd_ms": total_ms,
                        "train_step_ms": (step_end - start) / 1e6, "refresh_ms": (step_end - refresh_start) / 1e6})
    stored_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    owned_bytes = sum(p.numel() * p.element_size() for p in owned)
    memory = {"rank": rank, "weight_bytes": stored_bytes, "redundant_weight_bytes": stored_bytes - owned_bytes,
              "source_input_bytes": x.numel() * x.element_size(),
              "rss_peak_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if platform.system() == "Darwin" else 1024)}
    if distributed:
        gathered, memories = [None] * world, [None] * world
        dist.all_gather_object(gathered, raw)
        dist.all_gather_object(memories, memory)
        raw = [record for records in gathered for record in records]
    else:
        memories = [memory]
    metrics = empty_metrics()
    for key in metrics:
        maxima = [max(record[key] for record in raw if record["iteration"] == i) for i in range(samples)]
        values = torch.tensor(maxima, dtype=torch.float64)
        metrics[key] = {"p50": values.quantile(.5).item(), "p95": values.quantile(.95).item()}
    routing = trace.metadata["routing"]
    placement_counts = torch.zeros(world, dtype=torch.int64)
    if distributed:
        placement_counts += torch.bincount(model.destinations(ids), minlength=world)
        dist.all_reduce(placement_counts)
    else:
        placement_counts[0] = ids.numel()
    strategy = "fixed_replica_fixture" if replicas else "off"
    return {"id": f"{backend}/ep{world}/{trace.digest()[:16]}/{strategy}",
            "backend_id": backend, "forward_backend": backend, "backward_backend": backend,
            "kernel_family": "pytorch_cpu", "ep": world, "target_maxvio": routing["target_maxvio"],
            "strategy": "fixed_replica_fixture" if replicas else "off", "status": "pass",
            "reason": "CPU wall-clock regression only; native MoonEP/UltraEP not executed.",
            "model": asdict(config), "dtype": str(x.dtype), "trace_hash": trace.digest(),
            "activation_mode": "save", "topology": "single_host_cpu", "tokens_per_rank": int(mask.sum()),
            "load_balancing": bool(replicas), "baseline_id": None, "speedup": None,
            "redundant_budget": sum(len(ranks) for ranks in (replicas or {}).values()),
            "peak_hbm_delta_bytes": None,
            "routing": routing, "actual_maxvio": routing["actual_maxvio"], "placement": replicas,
            "scheduled_rank_counts": placement_counts.tolist(),
            "scheduled_rank_max_mean": placement_counts.max().item() / placement_counts.double().mean().item(),
            "correctness": correctness, "metrics": metrics, "raw": raw, "memory": memories,
            "peak_hbm_bytes": None, "hbm_reason": "CPU execution; no HBM measurement",
            "memory_scope": "Exact parameter/source input bytes and process high-water RSS; activation/communication workspace not instrumented.",
            "warmup": warmup, "samples": samples, "software": {"torch": torch.__version__, "python": platform.python_version()},
            "source_tokens_per_second": len(trace.x) / (metrics["fwd_bwd_ms"]["p50"] / 1000)}
