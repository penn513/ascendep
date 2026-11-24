"""Variable-split Gloo dispatcher with explicit replica training lifecycle."""
import torch
import torch.distributed as dist
from torch import nn

from .model import Expert, validate_inputs


class _Exchange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, send, recv, group):
        ctx.send, ctx.recv, ctx.group = send, recv, group
        out = x.new_empty((sum(recv),) + x.shape[1:])
        dist.all_to_all_single(out, x.contiguous(), recv, send, group=group)
        return out

    @staticmethod
    def backward(ctx, grad):
        out = grad.new_empty((sum(ctx.send),) + grad.shape[1:])
        dist.all_to_all_single(out, grad.contiguous(), ctx.send, ctx.recv, group=ctx.group)
        return out, None, None, None


class ExpertParallelMoE(nn.Module):
    """Rank-local tokens in/out; each rank stores only owned/replicated experts.

    All group ranks must call forward/backward, gradient return and refresh in
    the same order. No DDP averaging: the caller defines global loss scaling.
    replicas maps logical expert IDs to additional group-local ranks.
    """
    def __init__(self, config, group=None, replicas=None, dtype=torch.float64):
        super().__init__()
        if not dist.is_initialized():
            raise RuntimeError("initialize a Gloo process group first")
        if dist.get_backend(group) != "gloo":
            raise ValueError("CPU implementation requires Gloo")
        self.config, self.group = config, group
        self.rank, self.world = dist.get_rank(group), dist.get_world_size(group)
        if config.experts % self.world:
            raise ValueError("expert count must be divisible by EP size")
        self.replicas = {int(e): tuple(ranks) for e, ranks in (replicas or {}).items()}
        for e, ranks in self.replicas.items():
            if not 0 <= e < config.experts or len(set(ranks)) != len(ranks):
                raise ValueError("invalid or duplicate replica")
            if any(r < 0 or r >= self.world or r == self.owner(e) for r in ranks):
                raise ValueError("replica rank invalid or equals owner")
        local = [e for e in range(config.experts) if self.rank in self.locations(e)]
        self.experts = nn.ModuleDict({str(e): Expert(config, e, dtype) for e in local})

    def owner(self, e):
        return e // (self.config.experts // self.world)

    def locations(self, e):
        return (self.owner(e),) + self.replicas.get(e, ())

    def destinations(self, ids):
        flat = ids.reshape(-1)
        dest = flat // (self.config.experts // self.world)
        for e in self.replicas:
            idx = torch.where(flat == e)[0]
            locations = torch.tensor(self.locations(e), dtype=torch.int64)
            dest[idx] = locations[(idx + self.rank) % len(locations)]
        return dest

    def forward(self, x, ids, weights):
        validate_inputs(x, ids, weights, self.config)
        if x.device.type != "cpu":
            raise ValueError("CPU/Gloo path only")
        dest = self.destinations(ids)
        order = torch.argsort(dest, stable=True)
        send_counts = torch.bincount(dest, minlength=self.world)
        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts, group=self.group)
        send, recv = send_counts.tolist(), recv_counts.tolist()
        packed = x.repeat_interleave(self.config.top_k, dim=0)[order]
        # Even an empty receiver without active parameters must join backward
        # collectives when upstream hidden states do not require gradients.
        if torch.is_grad_enabled() and not packed.requires_grad:
            packed.requires_grad_()
        received = _Exchange.apply(packed, send, recv, self.group)
        received_ids = torch.empty(sum(recv), dtype=torch.int64)
        dist.all_to_all_single(received_ids, ids.reshape(-1)[order].contiguous(), recv, send, group=self.group)
        computed = received * 0
        for e, expert in self.experts.items():
            idx = torch.where(received_ids == int(e))[0]
            if idx.numel():
                computed = computed.index_add(0, idx, expert(received[idx]))
        returned = _Exchange.apply(computed, recv, send, self.group)
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(order.numel(), dtype=order.dtype)
        routes = returned[inverse].reshape(x.shape[0], self.config.top_k, self.config.hidden)
        return (routes * weights.unsqueeze(-1)).sum(dim=1)

    def return_replica_gradients(self):
        """Sum replica gradients onto owner exactly once, before optimizer.step."""
        for e in sorted(self.replicas):
            local = self.experts[str(e)] if str(e) in self.experts else None
            for name, shape in (("gate", (self.config.hidden, self.config.ffn)),
                                ("up", (self.config.hidden, self.config.ffn)),
                                ("down", (self.config.ffn, self.config.hidden))):
                param = getattr(local, name) if local is not None else None
                template = next(self.parameters())
                grad = torch.zeros(shape, dtype=template.dtype)
                if param is not None and param.grad is not None:
                    grad.copy_(param.grad)
                dist.all_reduce(grad, group=self.group)
                if param is not None:
                    param.grad = grad if self.rank == self.owner(e) else None

    @torch.no_grad()
    def refresh_replicas(self):
        """Broadcast updated owner weights; invoke after optimizer.step."""
        for e in sorted(self.replicas):
            local = self.experts[str(e)] if str(e) in self.experts else None
            for name, shape in (("gate", (self.config.hidden, self.config.ffn)),
                                ("up", (self.config.hidden, self.config.ffn)),
                                ("down", (self.config.ffn, self.config.hidden))):
                param = getattr(local, name) if local is not None else None
                tensor = param if param is not None else next(self.parameters()).new_empty(shape)
                src = self.owner(e) if self.group is None else dist.get_global_rank(self.group, self.owner(e))
                dist.broadcast(tensor, src=src, group=self.group)

    def owned_parameters(self):
        for e, expert in self.experts.items():
            if self.owner(int(e)) == self.rank:
                yield from expert.parameters()
