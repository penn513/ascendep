"""CPU reference and expert-parallel training layers; no optional NPU imports."""
from .model import MoEConfig, ReferenceMoE
from .distributed import ExpertParallelMoE

__all__ = ["MoEConfig", "ReferenceMoE", "ExpertParallelMoE"]
