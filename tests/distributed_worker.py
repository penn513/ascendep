import json
import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from ascendep import ExpertParallelMoE, MoEConfig
from ascendep.routing import make_trace
from ascendep.validation import validate_training


def main():
    torch.set_num_threads(1)
    dist.init_process_group("gloo", timeout=timedelta(seconds=35))
    try:
        results = {}
        for top_k in (1, 2):
            config = MoEConfig(experts=4, top_k=top_k, hidden=4, ffn=7)
            for scenario in ("balanced", "uneven", "zero_peers", "hotspot", "empty_source", "replica_migration"):
                trace = make_trace(config, tokens_per_rank=5, world=2)
                if scenario == "uneven":
                    trace.ids[:7] = torch.arange(top_k)
                elif scenario == "zero_peers":
                    trace.ids[:5] = torch.arange(top_k)
                    trace.ids[5:] = torch.arange(top_k) + 2
                elif scenario in ("hotspot", "replica_migration"):
                    trace.ids[:] = torch.arange(top_k)
                elif scenario == "empty_source":
                    trace.source_ranks[:] = 0
                second = make_trace(config, tokens_per_rank=5, world=2, seed=37)
                if scenario == "replica_migration":
                    second.ids[:] = torch.arange(top_k) + 2
                else:
                    second = trace
                replicas = {0: (1,)} if scenario == "replica_migration" else None
                if replicas:
                    results[f"top{top_k}/{scenario}/off"] = validate_training(config, [trace, second])
                results[f"top{top_k}/{scenario}"] = validate_training(config, [trace, second], replicas)
        # Frozen upstream inputs still require all ranks to participate in backward.
        config = MoEConfig(hidden=4, ffn=7)
        layer = ExpertParallelMoE(config)
        x = torch.ones(3, 4, dtype=torch.float64)
        ids = torch.tensor([[0, 1]] * 3)
        weights = torch.ones(3, 2, dtype=torch.float64)
        layer(x, ids, weights).sum().backward()
        if dist.get_rank() == 0:
            if "ASCENDEP_VALIDATION_REPORT" in os.environ:
                path = Path(os.environ["ASCENDEP_VALIDATION_REPORT"])
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(results, indent=2) + "\n")
            print("VALIDATION=" + json.dumps(results))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
