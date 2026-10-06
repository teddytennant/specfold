"""Folded bidirectional attention and the exact KV correction (eqs 6 and 8)."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def score_scale(d_head: int, dtype) -> Array:
    return jnp.reciprocal(jnp.sqrt(jnp.asarray(d_head, dtype=dtype)))


def streaming_attention_state(q: Array, k: Array, v: Array) -> tuple[Array, Array, Array]:
    """FlashAttention streaming-softmax state over a full key/value table (eq 6).

    Bidirectional: every query attends to every key. No causal mask.

    q: (queries, heads, d_head)
    k, v: (keys, heads, d_head)
    returns m (queries, heads), Z (queries, heads), O (queries, heads, d_head).
    O is the normalized output, so O * Z is the unnormalized accumulator.
    """
    if q.shape[0] == 0:
        heads, d_head = q.shape[1], q.shape[2]
        m = jnp.zeros((0, heads), dtype=q.dtype)
        z = jnp.zeros((0, heads), dtype=q.dtype)
        o = jnp.zeros((0, heads, d_head), dtype=q.dtype)
        return m, z, o
    scale = score_scale(q.shape[-1], q.dtype)
    scores = jnp.einsum("qhd,khd->qhk", q, k) * scale
    m = jnp.max(scores, axis=-1)
    exp_scores = jnp.exp(scores - m[..., None])
    z = jnp.sum(exp_scores, axis=-1)
    o = jnp.einsum("qhk,khd->qhd", exp_scores, v) / z[..., None]
    return m, z, o


def correct_attention(
    q: Array,
    k_parent: Array,
    v_parent: Array,
    k_child: Array,
    v_child: Array,
    m: Array,
    z: Array,
    o: Array,
    recompute_index: Array,
) -> tuple[Array, Array, Array]:
    """Convert a parent's streaming state into the child's (eq 8, Alg 2 lines 7-13).

    ``q`` is the inherited query (identical to the parent's query at folded
    positions). ``(m, z, o)`` is that query's exact softmax over the parent's
    concatenated prefix and block KV. Prefix entries are shared, so they stay
    inside the inherited state and are not passed here.

    The child's block KV differs from the parent's only at ``recompute_index``.
    Those positions are removed and replaced:

        m' = max(m, max_e q · k_child[e] / sqrt(d_h))
        f = exp(m - m')
        Z' = Z * f - sum_e a_old(e) + sum_e a_new(e)
        O' = (O * Z * f - sum_e a_old(e) v_parent[e] + sum_e a_new(e) v_child[e]) / Z'

    O is normalized, so O * Z * f is the unnormalized accumulator rescaled from
    the old max to m'. m' is allowed to stay above the true child max when the
    parent's max sat on a replaced key; the (m', Z', O') triple is still an
    exact softmax representation, which is what a later hop of the fold chain
    needs.

    An empty recompute set is the identity: the KV tables match.
    """
    recompute_index = jnp.asarray(recompute_index)
    if recompute_index.ndim != 1:
        raise ValueError("recompute_index must be a 1D index vector")
    if recompute_index.shape[0] == 0:
        return m, z, o

    k_old = k_parent[recompute_index]
    v_old = v_parent[recompute_index]
    k_new = k_child[recompute_index]
    v_new = v_child[recompute_index]

    scale = score_scale(q.shape[-1], q.dtype)
    s_old = jnp.einsum("fhd,rhd->fhr", q, k_old) * scale
    s_new = jnp.einsum("fhd,rhd->fhr", q, k_new) * scale

    m_prime = jnp.maximum(m, jnp.max(s_new, axis=-1))
    rescale = jnp.exp(m - m_prime)
    a_old = jnp.exp(s_old - m_prime[..., None])
    a_new = jnp.exp(s_new - m_prime[..., None])

    z_prime = z * rescale - jnp.sum(a_old, axis=-1) + jnp.sum(a_new, axis=-1)
    acc = o * z[..., None] * rescale[..., None]
    acc = acc - jnp.einsum("fhr,rhd->fhd", a_old, v_old)
    acc = acc + jnp.einsum("fhr,rhd->fhd", a_new, v_new)
    o_prime = acc / z_prime[..., None]
    return m_prime, z_prime, o_prime


def concat_prefix(prefix: Array | None, block: Array) -> Array:
    """Concatenate a shared prefix cache in front of a branch block table."""
    if prefix is None or prefix.shape[0] == 0:
        return block
    if prefix.shape[1:] != block.shape[1:]:
        raise ValueError(
            f"prefix shape {prefix.shape} is incompatible with block shape {block.shape}"
        )
    return jnp.concatenate([prefix, block], axis=0)
