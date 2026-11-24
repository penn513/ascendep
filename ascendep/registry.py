"""Planned native backends are inventory entries, never CPU aliases."""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Capability:
    backend_id: str
    repo: str
    implemented: bool = False
    device: str = "npu"
    variant_verified: bool = False
    commit: object = None
    pr_head_sha: object = None
    entrypoint: object = None
    forward: bool = False
    backward: bool = False
    routing_weight_gradient: bool = False
    kernel_family: object = None
    reason: str = "Native adapter and hardware validation pending; variants must be audited at a pinned commit."


BACKENDS = [
    Capability("cpu/reference", "local", True, "cpu", True, entrypoint="ascendep.model.ReferenceMoE", forward=True, backward=True, routing_weight_gradient=True, reason="Single process only; no NPU capability."),
    Capability("cpu/gloo", "local", True, "cpu", True, entrypoint="ascendep.distributed.ExpertParallelMoE", forward=True, backward=True, routing_weight_gradient=True, reason="Validated EP2 only; fixed replicas are a semantic fixture, not MoonEP/UltraEP."),
    Capability("megatron/pending_variants", "https://github.com/NVIDIA/Megatron-LM"),
    Capability("mindspeed/pending_variants", "https://gitcode.com/Ascend/MindSpeed"),
    Capability("fsdpturbo/pending_variants", "https://gitcode.com/Ascend/FSDPTurbo"),
    Capability("lueelu/megamoe", "https://github.com/lueelu/DeepEP/pull/10"),
    Capability("lueelu/deepep", "https://github.com/lueelu/DeepEP/pull/10"),
    Capability("lueelu/ep_gmm_fused", "https://github.com/lueelu/DeepEP/pull/10"),
    Capability("deepseek/deepep_ascend", "https://github.com/deepseek-ai/DeepEP-Ascend"),
    Capability("cann/deepep", "https://gitcode.com/cann/ops-transformer"),
    Capability("cann/megamoe", "https://gitcode.com/cann/ops-transformer"),
    Capability("cann/mc2/pending_variants", "https://gitcode.com/cann/ops-transformer"),
    Capability("moonep/pending_variants", "https://github.com/MoonshotAI/MoonEP"),
]


def capabilities():
    return [asdict(backend) for backend in BACKENDS]
