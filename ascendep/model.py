from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class MoEConfig:
    experts: int = 4
    top_k: int = 2
    hidden: int = 16
    ffn: int = 32
    seed: int = 17

    def __post_init__(self):
        if min(self.experts, self.hidden, self.ffn) < 1:
            raise ValueError("expert count and dimensions must be positive")
        if not 1 <= self.top_k <= self.experts:
            raise ValueError("require 1 <= top_k <= experts")


class Expert(nn.Module):
    def __init__(self, config, expert_id, dtype=torch.float64):
        super().__init__()
        gen = torch.Generator().manual_seed(config.seed + expert_id)
        self.gate = nn.Parameter(torch.randn(config.hidden, config.ffn, generator=gen, dtype=dtype) / config.hidden**0.5)
        self.up = nn.Parameter(torch.randn(config.hidden, config.ffn, generator=gen, dtype=dtype) / config.hidden**0.5)
        self.down = nn.Parameter(torch.randn(config.ffn, config.hidden, generator=gen, dtype=dtype) / config.ffn**0.5)

    def forward(self, x):
        return (F.silu(x @ self.gate) * (x @ self.up)) @ self.down


def validate_inputs(x, ids, weights, config):
    if x.ndim != 2 or x.shape[1] != config.hidden:
        raise ValueError("hidden states must have shape [tokens, hidden]")
    if ids.shape != (x.shape[0], config.top_k) or weights.shape != ids.shape:
        raise ValueError("routing must have shape [tokens, top_k]")
    if ids.dtype != torch.int64:
        raise ValueError("expert IDs must be int64")
    if ids.numel() and (ids.min() < 0 or ids.max() >= config.experts):
        raise ValueError("expert ID out of range")
    if config.top_k > 1 and (ids.sort(dim=1).values.diff(dim=1) == 0).any():
        raise ValueError("duplicate expert for a token")
    if x.dtype != weights.dtype or x.device != weights.device or x.device != ids.device:
        raise ValueError("input and routing dtype/device mismatch")
    if not torch.isfinite(x).all() or not torch.isfinite(weights).all():
        raise ValueError("non-finite input or routing weight")


class ReferenceMoE(nn.Module):
    """Independent logical-expert loop; no dispatcher or placement code."""
    def __init__(self, config=MoEConfig(), dtype=torch.float64):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleDict({str(e): Expert(config, e, dtype) for e in range(config.experts)})

    def forward(self, x, ids, weights):
        validate_inputs(x, ids, weights, self.config)
        out = x * 0 + weights.sum() * 0
        for e, expert in self.experts.items():
            token, slot = torch.where(ids == int(e))
            if token.numel():
                out = out.index_add(0, token, expert(x[token]) * weights[token, slot, None])
        return out


class LatentMoE(nn.Module):
    """Project model hidden -> EP latent dimension -> model hidden."""
    def __init__(self, model_hidden, moe):
        super().__init__()
        self.moe = moe
        param = next(moe.parameters())
        self.project_in = nn.Linear(model_hidden, moe.config.hidden, bias=False, dtype=param.dtype)
        self.project_out = nn.Linear(moe.config.hidden, model_hidden, bias=False, dtype=param.dtype)

    def forward(self, x, ids, weights):
        return self.project_out(self.moe(self.project_in(x), ids, weights))
