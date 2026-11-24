"""CANN BF16 forward integration; the runner also checks backward availability."""
import torch.distributed as dist
from cann_ops_transformer.ops import get_symm_buffer_for_mega_moe, mega_moe
from ep.runner import run


def forward(layer, x, ids, weights):
    if not hasattr(layer, "buffer"):
        layer.buffer = get_symm_buffer_for_mega_moe(dist.group.WORLD, num_experts=24,
            num_max_tokens_per_rank=len(x), num_topk=6, hidden=x.shape[1],
            intermediate_hidden=layer.down_proj.shape[-1], dispatch_quant_mode=0,
            dispatch_quant_out_dtype=None)
    w1 = [w.T.contiguous() for w in layer.gate_up_proj]
    w2 = [w.T.contiguous() for w in layer.down_proj]
    y, _ = mega_moe(x, ids.int(), weights.float(), w1, w2, layer.buffer)
    return y


if __name__ == "__main__":
    run(forward, "megamoe_cann")
