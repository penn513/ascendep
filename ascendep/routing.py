"""Frozen, integrity-checked traces; routing generation is outside timing."""
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from math import isfinite

import torch

from .model import validate_inputs


def routing_stats(ids, experts, world=1):
    counts = torch.bincount(ids.reshape(-1), minlength=experts)
    mean = counts.double().mean().item()
    rank_counts = counts.reshape(world, -1).sum(dim=1).double()
    return {"counts": counts.tolist(), "assignments": int(counts.sum()),
            "actual_maxvio": counts.max().item() / mean - 1 if mean else 0.,
            "cv": counts.double().std(unbiased=False).item() / mean if mean else 0.,
            "owner_rank_max_mean": rank_counts.max().item() / rank_counts.mean().item() if mean else 0.}


def make_routing(tokens, experts, top_k, target=0., seed=31):
    if tokens < 1 or not 1 <= top_k <= experts or not isfinite(target) or target < 0:
        raise ValueError("invalid routing dimensions or target")
    upper = experts / top_k - 1
    tolerance = max(.02 * target, .005)
    if target == 0:
        ids = torch.arange(tokens * top_k).reshape(tokens, top_k) % experts
        stats = routing_stats(ids, experts)
        return ids, dict(stats, target_maxvio=target, sigma=0., error=stats["actual_maxvio"],
                         tolerance=tolerance, status="matched" if stats["actual_maxvio"] <= tolerance else "discrete_limit")
    z = torch.randn(experts, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
    best = None
    lo, hi = 0., 64.
    # Same random seed for each candidate reduces sampling noise during search.
    for iteration in range(36):
        sigma = 0. if iteration == 0 else (lo + hi) / 2
        probabilities = torch.softmax(sigma * z, dim=0)
        generator = torch.Generator().manual_seed(seed + 1)
        chunks = [torch.multinomial(probabilities.expand(min(1024, tokens - start), -1), top_k,
                                    replacement=False, generator=generator) for start in range(0, tokens, 1024)]
        ids = torch.cat(chunks)
        stats = routing_stats(ids, experts)
        error = abs(stats["actual_maxvio"] - target)
        if best is None or error < best[0]:
            best = error, ids, stats, sigma
        if error <= tolerance:
            break
        if stats["actual_maxvio"] < target:
            lo = sigma
        else:
            hi = sigma
    error, ids, stats, sigma = best
    status = "unreachable" if target > upper else "matched" if error <= tolerance else "not_matched"
    return ids, dict(stats, target_maxvio=target, sigma=sigma, error=error, tolerance=tolerance, status=status)


@dataclass
class Trace:
    x: torch.Tensor
    ids: torch.Tensor
    weights: torch.Tensor
    source_ranks: torch.Tensor
    metadata: dict

    def digest(self):
        import json
        h = sha256(json.dumps(self.metadata, sort_keys=True, allow_nan=False).encode())
        for tensor in (self.x, self.ids, self.weights, self.source_ranks):
            h.update(str((tuple(tensor.shape), str(tensor.dtype))).encode())
            h.update(tensor.detach().contiguous().view(torch.uint8).numpy().tobytes())
        return h.hexdigest()

    def validate(self, config):
        validate_inputs(self.x, self.ids, self.weights, config)
        if self.source_ranks.shape != (len(self.x),) or self.source_ranks.dtype != torch.int64:
            raise ValueError("invalid source rank layout")
        world = self.metadata["world"]
        if self.source_ranks.numel() and (self.source_ranks.min() < 0 or self.source_ranks.max() >= world):
            raise ValueError("source rank out of range")

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(x=self.x, ids=self.ids, weights=self.weights, source_ranks=self.source_ranks,
                        metadata=self.metadata, digest=self.digest()), path)

    @classmethod
    def load(cls, path):
        data = torch.load(path, map_location="cpu", weights_only=True)
        digest = data.pop("digest")
        trace = cls(**data)
        if trace.digest() != digest:
            raise ValueError("trace content hash mismatch")
        return trace


def make_trace(config, tokens_per_rank=8, world=1, target=0., seed=31):
    if world < 1 or config.experts % world:
        raise ValueError("experts must be divisible by world size")
    tokens = tokens_per_rank * world
    ids, stats = make_routing(tokens, config.experts, config.top_k, target, seed)
    gen = torch.Generator().manual_seed(seed + 2)
    x = torch.randn(tokens, config.hidden, dtype=torch.float64, generator=gen)
    weights = torch.rand(tokens, config.top_k, dtype=torch.float64, generator=gen)
    weights /= weights.sum(dim=1, keepdim=True)
    stats.update(routing_stats(ids, config.experts, world))
    return Trace(x, ids, weights, torch.arange(world).repeat_interleave(tokens_per_rank),
                 {"generator_version": 2, "sampling_chunk_size": 1024, "seed": seed, "world": world, "routing": stats})


def stream_route_summary(experts, top_k, tokens_per_rank, world, target=0., seed=31, chunk_size=1024):
    """Audit large EP routes without model weights, hidden states or dense traces.

    Only a chunk's [tokens, experts] sampling workspace exists at once. The
    stream has its own generator version; it does not claim trace-v1 identity.
    """
    if min(tokens_per_rank, world, experts, chunk_size) < 1 or experts % world or not 1 <= top_k <= experts or not isfinite(target) or target < 0:
        raise ValueError("invalid streaming routing configuration")
    tokens = tokens_per_rank * world
    z = torch.randn(experts, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
    tolerance, upper = max(.02 * target, .005), experts / top_k - 1

    def candidate(sigma):
        counts = torch.zeros(experts, dtype=torch.int64)
        generator = torch.Generator().manual_seed(seed + 1)
        probabilities = torch.softmax(sigma * z, dim=0)
        digest = sha256()
        for start in range(0, tokens, chunk_size):
            size = min(tokens - start, chunk_size)
            if target == 0:
                ids = torch.arange(start * top_k, (start + size) * top_k).reshape(size, top_k) % experts
            else:
                ids = torch.multinomial(probabilities.expand(size, -1), top_k, replacement=False, generator=generator)
            if top_k > 1 and (ids.sort(dim=1).values.diff(dim=1) == 0).any():
                raise AssertionError("duplicate expert in streamed routing")
            counts += torch.bincount(ids.reshape(-1), minlength=experts)
            digest.update(ids.contiguous().numpy().tobytes())
        mean = tokens * top_k / experts
        actual = counts.max().item() / mean - 1
        rank_counts = counts.reshape(world, -1).sum(dim=1)
        assert counts.sum().item() == tokens * top_k
        return {"counts": counts.tolist(), "assignments": tokens * top_k, "sigma": sigma,
                "actual_maxvio": actual, "error": abs(actual - target), "route_ids_hash": digest.hexdigest(),
                "cv": counts.double().std(unbiased=False).item() / mean,
                "owner_rank_max_mean": rank_counts.max().item() / (tokens * top_k / world)}

    best, lo, hi = None, 0., 64.
    for iteration in range(1 if target == 0 or target > upper else 30):
        sigma = 64. if target > upper else 0. if iteration == 0 else (lo + hi) / 2
        stats = candidate(sigma)
        if best is None or stats["error"] < best["error"]:
            best = stats
        if stats["error"] <= tolerance:
            break
        if stats["actual_maxvio"] < target:
            lo = sigma
        else:
            hi = sigma
    status = "unreachable" if target > upper else "matched" if best["error"] <= tolerance else "not_matched"
    return dict(best, status=status, target_maxvio=target, tolerance=tolerance, experts=experts,
                top_k=top_k, tokens_per_rank=tokens_per_rank, world=world, seed=seed,
                generator_version="stream-v1", chunk_size=chunk_size, model_weights_allocated=False)
