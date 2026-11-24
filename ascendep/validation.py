"""Numerical oracle shared by CLI and real multiprocess regression."""
import torch
import torch.distributed as dist

from .distributed import ExpertParallelMoE
from .model import ReferenceMoE


def validate_training(config, traces, replicas=None, atol=1e-10, rtol=1e-9):
    rank = dist.get_rank()
    model = ExpertParallelMoE(config, replicas=replicas)
    reference = ReferenceMoE(config)
    optimizer = torch.optim.SGD(model.owned_parameters(), lr=.03)
    ref_optimizer = torch.optim.SGD(reference.parameters(), lr=.03)
    errors = {key: 0. for key in ("output", "dx", "dtopk_weight", "expert_gradient", "updated_weight")}

    def check(key, actual, expected):
        if actual is None:
            actual = torch.zeros_like(expected)
        if expected is None:
            expected = torch.zeros_like(actual)
        error = (actual - expected).abs().max().item() if actual.numel() else 0.
        errors[key] = max(errors[key], error)
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)

    for trace in traces:
        trace.validate(config)
        optimizer.zero_grad(set_to_none=True)
        model.zero_grad(set_to_none=True)
        ref_optimizer.zero_grad(set_to_none=True)
        mask = trace.source_ranks == rank
        x = trace.x[mask].clone().requires_grad_()
        weights = trace.weights[mask].clone().requires_grad_()
        rx = trace.x.clone().requires_grad_()
        rw = trace.weights.clone().requires_grad_()
        grad = torch.linspace(-.7, .9, trace.x.numel(), dtype=trace.x.dtype).reshape_as(trace.x) / max(1, trace.x.numel())
        expected = reference(rx, trace.ids, rw)
        expected.backward(grad)
        actual = model(x, trace.ids[mask], weights)
        check("output", actual, expected[mask])
        actual.backward(grad[mask])
        model.return_replica_gradients()
        check("dx", x.grad, rx.grad[mask])
        check("dtopk_weight", weights.grad, rw.grad[mask])
        for e, expert in model.experts.items():
            if model.owner(int(e)) == rank:
                for name, param in expert.named_parameters():
                    ref_param = getattr(reference.experts[e], name)
                    a = param.grad if param.grad is not None else torch.zeros_like(param)
                    b = ref_param.grad if ref_param.grad is not None else torch.zeros_like(ref_param)
                    check("expert_gradient", a, b)
        optimizer.step()
        ref_optimizer.step()
        model.refresh_replicas()
        for e, expert in model.experts.items():
            for name, param in expert.named_parameters():
                check("updated_weight", param, getattr(reference.experts[e], name))
    values = torch.tensor(list(errors.values()), dtype=torch.float64)
    dist.all_reduce(values, op=dist.ReduceOp.MAX)
    return {"status": "pass", "steps": len(traces), "atol": atol, "rtol": rtol,
            "max_abs_error": dict(zip(errors, values.tolist()))}
