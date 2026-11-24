"""Run: python -m ascendep launch examples/train.py"""
from datetime import timedelta

import torch
import torch.distributed as dist

from ascendep import ExpertParallelMoE, MoEConfig
from ascendep.model import LatentMoE
from ascendep.routing import make_trace


def main():
    torch.set_num_threads(1)
    dist.init_process_group("gloo", timeout=timedelta(seconds=60))
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        config = MoEConfig()
        trace = make_trace(config, world=world)
        local = trace.source_ranks == rank
        # Load checkpoint on owners, then refresh replicas before first forward.
        ep = ExpertParallelMoE(config, replicas={0: (1,)} if world == 2 else None)
        ep.refresh_replicas()
        torch.manual_seed(7)  # Replicated projections start identically on all ranks.
        layer = LatentMoE(config.hidden, ep)
        projection_parameters = list(layer.project_in.parameters()) + list(layer.project_out.parameters())
        optimizer = torch.optim.SGD(list(ep.owned_parameters()) + projection_parameters, lr=.01)
        for step in range(2):
            layer.zero_grad(set_to_none=True)
            x = trace.x[local].clone().requires_grad_()
            weights = trace.weights[local].clone().requires_grad_()
            output = layer(x, trace.ids[local], weights)
            # Each rank contributes its local sum divided by GLOBAL element count.
            loss = output.square().sum() / trace.x.numel()
            loss.backward()
            ep.return_replica_gradients()
            # Dense projections are replicated, unlike EP expert parameters.
            for param in projection_parameters:
                dist.all_reduce(param.grad)
            optimizer.step()
            ep.refresh_replicas()
            global_loss = loss.detach().clone()
            dist.all_reduce(global_loss)
            if rank == 0:
                print(f"step={step} global_loss={global_loss.item():.8f}")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
