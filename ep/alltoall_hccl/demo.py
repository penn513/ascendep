"""Variable-split HCCL dispatch, rank-local SwiGLU experts, inverse combine."""
import torch
import torch.distributed as dist
from torch.nn import functional as F


class Exchange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, send, recv):
        ctx.send, ctx.recv = send, recv
        out = x.new_empty((sum(recv), *x.shape[1:]))
        dist.all_to_all_single(out, x.contiguous(), recv, send)
        return out

    @staticmethod
    def backward(ctx, grad):
        return Exchange.apply(grad, ctx.recv, ctx.send), None, None


def forward(layer, x, ids, weights):
    expert = ids.flatten().long()
    order = expert.argsort(stable=True)
    world = dist.get_world_size()
    local_experts = 24 // world
    counts = torch.bincount(expert, minlength=24).reshape(world, local_experts)
    peer_counts = torch.empty_like(counts)
    dist.all_to_all_single(peer_counts, counts)
    send = counts.sum(1).tolist()
    recv = peer_counts.sum(1).tolist()
    packed = Exchange.apply(x[order // 6], send, recv)
    local = torch.arange(local_experts, device=x.device).repeat(world).repeat_interleave(peer_counts.flatten())
    perm = local.argsort(stable=True)
    chunks = packed[perm].split(peer_counts.sum(0).tolist())
    values = []
    for i, chunk in enumerate(chunks):
        gate, up = F.linear(chunk, layer.gate_up_proj[i]).chunk(2, -1)
        values.append(F.linear(F.silu(gate.clamp(max=10)) * up.clamp(-10, 10), layer.down_proj[i]))
    returned = Exchange.apply(torch.cat(values)[perm.argsort()], recv, send)
    return (returned[order.argsort()].reshape(x.shape[0], 6, -1) * weights.unsqueeze(-1)).sum(1)


if __name__ == "__main__":
    from ep.runner import run
    run(forward, "alltoall_hccl")
