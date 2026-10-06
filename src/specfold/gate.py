"""Relative-residual gate (eqs 4-5, Algorithm 1 line 5)."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def relative_residual(h_child: Array, h_parent: Array, eps: float | None = None) -> Array:
    """L2 relative residual rho = ||h_child - h_parent||_2 / ||h_parent||_2.

    Equation 4 writes this ratio of norms. A parent norm at or below ``eps``
    (default: the dtype's smallest positive normal) is treated as +inf so the
    position recomputes and the 0/0 case cannot produce NaN. Algorithm 1's
    cross-multiplied form is undefined as a ratio there; +inf matches the
    brief and is the exact direction (recompute).
    """
    diff = jnp.linalg.norm(h_child - h_parent, axis=-1)
    parent_norm = jnp.linalg.norm(h_parent, axis=-1)
    if eps is None:
        eps = float(jnp.finfo(parent_norm.dtype).tiny)
    safe = parent_norm > eps
    denom = jnp.where(safe, parent_norm, jnp.ones_like(parent_norm))
    inf = jnp.full_like(parent_norm, jnp.inf)
    return jnp.where(safe, diff / denom, inf)


def recompute_mask(
    h_child: Array,
    h_parent: Array,
    delta: float,
    eps: float | None = None,
) -> Array:
    """True where the position must be recomputed (rho > delta). Fold on rho <= delta."""
    return relative_residual(h_child, h_parent, eps=eps) > delta


def branch_recompute_mask(
    hidden: Array,
    parent: Array,
    delta: float,
    eps: float | None = None,
) -> Array:
    """Per-branch recompute mask, shape (branches, positions).

    Branch 0 is the main branch and always recomputes (Algorithm 1 line 3).
    Any other branch with a negative parent index is treated the same way.
    ``parent[i]`` is the parent branch index, or negative if ``i`` has none.
    """
    hidden = jnp.asarray(hidden)
    branches, positions, _ = hidden.shape
    parent_list = [int(p) for p in list(parent)]
    if len(parent_list) != branches:
        raise ValueError(
            f"parent length {len(parent_list)} does not match branches {branches}"
        )
    mask = jnp.ones((branches, positions), dtype=bool)
    for i, p in enumerate(parent_list):
        if i == 0 or p < 0:
            continue
        if p >= branches:
            raise ValueError(f"parent[{i}]={p} is out of range for {branches} branches")
        mask = mask.at[i].set(
            recompute_mask(hidden[i], hidden[p], delta, eps=eps)
        )
    return mask
