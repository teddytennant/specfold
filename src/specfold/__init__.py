"""SpecFold folded transformer layer (arXiv:2610.04875), algorithmic core."""

from specfold.attention import correct_attention, streaming_attention_state
from specfold.gate import branch_recompute_mask, recompute_mask, relative_residual
from specfold.layer import (
    LayerResult,
    ProjectionTrace,
    SpecFoldLayer,
    SpecFoldWeights,
    apply_rope,
    ffn_from_hidden,
    qkv_from_hidden,
    rms_norm,
    specfold_layer,
    specfold_stack,
    swiglu,
    topological_branches,
)

__all__ = [
    "LayerResult",
    "ProjectionTrace",
    "SpecFoldLayer",
    "SpecFoldWeights",
    "apply_rope",
    "branch_recompute_mask",
    "correct_attention",
    "ffn_from_hidden",
    "qkv_from_hidden",
    "recompute_mask",
    "relative_residual",
    "rms_norm",
    "specfold_layer",
    "specfold_stack",
    "streaming_attention_state",
    "swiglu",
    "topological_branches",
]
