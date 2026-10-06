"""SpecFoldLayer: one folded pre-norm transformer block (Algorithm 2)."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array

from specfold.attention import (
    concat_prefix,
    correct_attention,
    streaming_attention_state,
)
from specfold.gate import branch_recompute_mask


@dataclass(frozen=True)
class SpecFoldWeights:
    """One layer's weights. Linear maps use row-vector layout, y = x @ W.

    w_qkv: (d_model, 3 * n_heads * d_head), packed as Q then K then V.
    w_o: (n_heads * d_head, d_model)
    SwiGLU: silu(x @ w_gate) * (x @ w_up) @ w_down.
    """

    attn_norm: Array
    ffn_norm: Array
    w_qkv: Array
    w_o: Array
    w_gate: Array
    w_up: Array
    w_down: Array
    n_heads: int
    rope_theta: float = 10_000.0
    rms_eps: float = 1e-6

    @property
    def d_model(self) -> int:
        return int(self.attn_norm.shape[0])

    @property
    def d_head(self) -> int:
        return int(self.w_qkv.shape[-1] // (3 * self.n_heads))

    @property
    def d_ff(self) -> int:
        return int(self.w_gate.shape[-1])


class ProjectionTrace:
    """Rows actually presented to the QKV and FFN projections.

    A folded position that is still projected shows up here. The matmuls live
    only in ``qkv_from_hidden`` and ``ffn_from_hidden``, and those functions
    record every row they see.
    """

    def __init__(self) -> None:
        self.qkv_rows: list[tuple[int, int]] = []
        self.ffn_rows: list[tuple[int, int]] = []

    def record_qkv(self, branch: int, positions) -> None:
        for pos in positions:
            self.qkv_rows.append((int(branch), int(pos)))

    def record_ffn(self, branch: int, positions) -> None:
        for pos in positions:
            self.ffn_rows.append((int(branch), int(pos)))


@dataclass(frozen=True)
class LayerResult:
    hidden: Array
    h_plus: Array
    q: Array
    k: Array
    v: Array
    m: Array
    z: Array
    o: Array
    ffn: Array


def rms_norm(x: Array, weight: Array, eps: float = 1e-6) -> Array:
    """Llama-style RMSNorm. ``weight`` is a learned or fixed per-channel scale."""
    var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    return x * jax.lax.rsqrt(var + eps) * weight


def apply_rope(x: Array, positions: Array, theta: float = 10_000.0) -> Array:
    """Rotary embedding on the last axis. ``x`` is (positions, heads, d_head).

    Applied to Q and K only. The paper writes RoPE(W_QKV(...)) as the QKV
    triple; V is the unrotated value projection, as in the usual rotary block.
    """
    d_head = x.shape[-1]
    if d_head % 2 != 0:
        raise ValueError(f"RoPE requires an even head dim, got {d_head}")
    half = d_head // 2
    idx = jnp.arange(half, dtype=x.dtype)
    inv_freq = 1.0 / (theta ** (idx / d_head))
    angles = positions.astype(x.dtype)[:, None] * inv_freq[None, :]
    cos = jnp.cos(angles)[:, None, :]
    sin = jnp.sin(angles)[:, None, :]
    x1 = x[..., :half]
    x2 = x[..., half:]
    return jnp.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)


def swiglu(x: Array, w_gate: Array, w_up: Array, w_down: Array) -> Array:
    """Gate-up SiLU FFN (SwiGLU). ``x`` is (rows, d_model)."""
    return (jax.nn.silu(x @ w_gate) * (x @ w_up)) @ w_down


def qkv_from_hidden(
    h_rows: Array,
    positions: Array,
    weights: SpecFoldWeights,
    trace: ProjectionTrace | None = None,
    branch: int | None = None,
    position_index: Array | None = None,
) -> tuple[Array, Array, Array]:
    """RMSNorm, packed QKV, and RoPE on Q and K, for the rows passed in only."""
    if trace is not None:
        if branch is None or position_index is None:
            raise ValueError("a trace requires branch and position_index")
        trace.record_qkv(branch, position_index)
    x = rms_norm(h_rows, weights.attn_norm, weights.rms_eps)
    packed = x @ weights.w_qkv
    rows = packed.shape[0]
    packed = packed.reshape(rows, 3, weights.n_heads, weights.d_head)
    q = apply_rope(packed[:, 0], positions, weights.rope_theta)
    k = apply_rope(packed[:, 1], positions, weights.rope_theta)
    v = packed[:, 2]
    return q, k, v


def ffn_from_hidden(
    h_rows: Array,
    weights: SpecFoldWeights,
    trace: ProjectionTrace | None = None,
    branch: int | None = None,
    position_index: Array | None = None,
) -> Array:
    """RMSNorm and SwiGLU, for the rows passed in only."""
    if trace is not None:
        if branch is None or position_index is None:
            raise ValueError("a trace requires branch and position_index")
        trace.record_ffn(branch, position_index)
    return swiglu(
        rms_norm(h_rows, weights.ffn_norm, weights.rms_eps),
        weights.w_gate,
        weights.w_up,
        weights.w_down,
    )


def attention_output_proj(o: Array, w_o: Array) -> Array:
    """Concatenate heads and apply W_O. ``o`` is (positions, heads, d_head)."""
    positions, heads, d_head = o.shape
    return o.reshape(positions, heads * d_head) @ w_o


def topological_branches(parent) -> list[int]:
    """Parents before children. A cycle raises ValueError."""
    parent_list = [int(p) for p in list(parent)]
    branches = len(parent_list)
    children: list[list[int]] = [[] for _ in range(branches)]
    indeg = [0] * branches
    for i, p in enumerate(parent_list):
        if p < 0:
            continue
        if p >= branches:
            raise ValueError(f"parent[{i}]={p} is out of range for {branches} branches")
        children[p].append(i)
        indeg[i] += 1
    queue = [i for i in range(branches) if indeg[i] == 0]
    order: list[int] = []
    while queue:
        i = queue.pop(0)
        order.append(i)
        for child in children[i]:
            indeg[child] -= 1
            if indeg[child] == 0:
                queue.append(child)
    if len(order) != branches:
        raise ValueError("parent map contains a cycle")
    return order


def block_positions(n_positions: int, prefix_len: int, dtype) -> Array:
    """Absolute positions of the block, starting after a prefix of length P."""
    return jnp.arange(prefix_len, prefix_len + n_positions, dtype=dtype)


def _validate_weights(weights: SpecFoldWeights, d_model: int) -> None:
    heads = weights.n_heads
    if weights.attn_norm.shape != (d_model,):
        raise ValueError("attn_norm shape does not match d_model")
    if weights.ffn_norm.shape != (d_model,):
        raise ValueError("ffn_norm shape does not match d_model")
    if weights.w_qkv.shape[0] != d_model or weights.w_qkv.shape[1] % (3 * heads) != 0:
        raise ValueError("w_qkv must be (d_model, 3 * n_heads * d_head)")
    d_head = weights.d_head
    if weights.w_o.shape != (heads * d_head, d_model):
        raise ValueError("w_o must be (n_heads * d_head, d_model)")
    if weights.w_gate.shape[0] != d_model or weights.w_up.shape != weights.w_gate.shape:
        raise ValueError("w_gate and w_up must be (d_model, d_ff)")
    if weights.w_down.shape != (weights.d_ff, d_model):
        raise ValueError("w_down must be (d_ff, d_model)")


def specfold_layer(
    hidden: Array,
    parent,
    recompute: Array,
    weights: SpecFoldWeights,
    prefix_k: Array | None = None,
    prefix_v: Array | None = None,
    trace: ProjectionTrace | None = None,
) -> LayerResult:
    """One SpecFold layer (Algorithm 2).

    hidden: (branches, positions, d_model). Hidden states are never overwritten
    with parent copies.
    parent: parent branch index per branch, negative if the branch has none.
    recompute: bool mask (branches, positions). True means the position is in
    R_i and is projected. False means it is folded.
    prefix_k, prefix_v: shared RoPE-transformed prefix cache, (P, heads, d_head).
    P = 0 is allowed (pass None).

    Branches are visited in topological order. Each branch inherits the parent's
    Q, K, V, streaming state (m, Z, O), and FFN table, then overwrites only
    recomputed positions. Folded queries keep the inherited Q, K, V and correct
    the inherited attention state against the immediate parent (eq 8). That is
    exact along a fold chain because the parent tables are already exact.

    Algorithm 2 writes the FlashAttention call inside the recompute loop. The
    KV table has to be complete before any query is attended, so all recomputed
    Q, K, V are scattered first, then attention runs. That is the order of the
    Triton pipeline (ln_qkv_rope_scatter, then flash_attn).
    """
    hidden = jnp.asarray(hidden)
    if hidden.ndim != 3:
        raise ValueError("hidden must have shape (branches, positions, d_model)")
    branches, n_positions, d_model = hidden.shape
    recompute = jnp.asarray(recompute, dtype=bool)
    if recompute.shape != (branches, n_positions):
        raise ValueError(
            f"recompute shape {recompute.shape} != {(branches, n_positions)}"
        )
    _validate_weights(weights, d_model)
    parent_list = [int(p) for p in list(parent)]
    if len(parent_list) != branches:
        raise ValueError("parent length does not match the branch count")
    order = topological_branches(parent_list)

    heads, d_head = weights.n_heads, weights.d_head
    dtype = hidden.dtype
    if prefix_k is None and prefix_v is None:
        prefix_k = jnp.zeros((0, heads, d_head), dtype=dtype)
        prefix_v = jnp.zeros((0, heads, d_head), dtype=dtype)
    elif prefix_k is None or prefix_v is None:
        raise ValueError("prefix_k and prefix_v must both be set or both be None")
    else:
        prefix_k = jnp.asarray(prefix_k, dtype=dtype)
        prefix_v = jnp.asarray(prefix_v, dtype=dtype)
        if prefix_k.shape != prefix_v.shape:
            raise ValueError("prefix_k and prefix_v shapes differ")
        if prefix_k.ndim != 3 or prefix_k.shape[1:] != (heads, d_head):
            raise ValueError(
                f"prefix cache must have shape (P, {heads}, {d_head}), got {prefix_k.shape}"
            )
    positions = block_positions(n_positions, int(prefix_k.shape[0]), dtype)

    q = jnp.zeros((branches, n_positions, heads, d_head), dtype=dtype)
    k = jnp.zeros_like(q)
    v = jnp.zeros_like(q)
    m = jnp.zeros((branches, n_positions, heads), dtype=dtype)
    z = jnp.zeros_like(m)
    o = jnp.zeros_like(q)
    ffn = jnp.zeros((branches, n_positions, d_model), dtype=dtype)
    h_plus = jnp.zeros_like(hidden)
    h_out = jnp.zeros_like(hidden)

    for i in order:
        p = parent_list[i]
        idx = jnp.where(recompute[i])[0]
        folded = jnp.where(~recompute[i])[0]
        if p < 0:
            if folded.shape[0] != 0:
                raise ValueError(
                    f"branch {i} has no parent but folds positions {folded.tolist()}"
                )
            qi = jnp.zeros((n_positions, heads, d_head), dtype=dtype)
            ki = jnp.zeros_like(qi)
            vi = jnp.zeros_like(qi)
            mi = jnp.zeros((n_positions, heads), dtype=dtype)
            zi = jnp.zeros_like(mi)
            oi = jnp.zeros_like(qi)
            fi = jnp.zeros((n_positions, d_model), dtype=dtype)
            k_parent = ki
            v_parent = vi
        else:
            qi = q[p]
            ki = k[p]
            vi = v[p]
            mi = m[p]
            zi = z[p]
            oi = o[p]
            fi = ffn[p]
            k_parent = ki
            v_parent = vi

        if idx.shape[0] != 0:
            q_rows, k_rows, v_rows = qkv_from_hidden(
                hidden[i, idx],
                positions[idx],
                weights,
                trace=trace,
                branch=i,
                position_index=idx,
            )
            qi = qi.at[idx].set(q_rows)
            ki = ki.at[idx].set(k_rows)
            vi = vi.at[idx].set(v_rows)
            m_rows, z_rows, o_rows = streaming_attention_state(
                qi[idx],
                concat_prefix(prefix_k, ki),
                concat_prefix(prefix_v, vi),
            )
            mi = mi.at[idx].set(m_rows)
            zi = zi.at[idx].set(z_rows)
            oi = oi.at[idx].set(o_rows)

        if p >= 0 and folded.shape[0] != 0 and idx.shape[0] != 0:
            m_corr, z_corr, o_corr = correct_attention(
                qi[folded],
                k_parent,
                v_parent,
                ki,
                vi,
                mi[folded],
                zi[folded],
                oi[folded],
                idx,
            )
            mi = mi.at[folded].set(m_corr)
            zi = zi.at[folded].set(z_corr)
            oi = oi.at[folded].set(o_corr)

        # Residual is never folded (eq 9): the branch's own hidden state.
        h_plus_i = hidden[i] + attention_output_proj(oi, weights.w_o)
        if idx.shape[0] != 0:
            f_rows = ffn_from_hidden(
                h_plus_i[idx],
                weights,
                trace=trace,
                branch=i,
                position_index=idx,
            )
            fi = fi.at[idx].set(f_rows)
        h_out_i = h_plus_i + fi

        q = q.at[i].set(qi)
        k = k.at[i].set(ki)
        v = v.at[i].set(vi)
        m = m.at[i].set(mi)
        z = z.at[i].set(zi)
        o = o.at[i].set(oi)
        ffn = ffn.at[i].set(fi)
        h_plus = h_plus.at[i].set(h_plus_i)
        h_out = h_out.at[i].set(h_out_i)

    return LayerResult(
        hidden=h_out,
        h_plus=h_plus,
        q=q,
        k=k,
        v=v,
        m=m,
        z=z,
        o=o,
        ffn=ffn,
    )


class SpecFoldLayer:
    """Algorithm 2 as a callable layer holding weights and an optional trace."""

    def __init__(
        self,
        weights: SpecFoldWeights,
        trace: ProjectionTrace | None = None,
    ) -> None:
        self.weights = weights
        self.trace = trace

    def __call__(
        self,
        hidden: Array,
        parent,
        recompute: Array,
        prefix_k: Array | None = None,
        prefix_v: Array | None = None,
    ) -> LayerResult:
        return specfold_layer(
            hidden,
            parent,
            recompute,
            self.weights,
            prefix_k=prefix_k,
            prefix_v=prefix_v,
            trace=self.trace,
        )


def specfold_stack(
    hidden: Array,
    parent,
    layers: list[SpecFoldWeights],
    delta: float,
    prefixes: list[tuple[Array | None, Array | None]] | None = None,
    traces: list[ProjectionTrace | None] | None = None,
) -> tuple[Array, list[LayerResult], list[Array]]:
    """Algorithm 1's layer loop, without the embedding or the LM head.

    The gate is recomputed from each layer's input hidden states. Returns
    the final hidden state, each layer's ``LayerResult``, and each gate mask.
    """
    if prefixes is None:
        prefixes = [(None, None)] * len(layers)
    if traces is None:
        traces = [None] * len(layers)
    if len(prefixes) != len(layers) or len(traces) != len(layers):
        raise ValueError("prefixes and traces must match the number of layers")
    h = hidden
    results: list[LayerResult] = []
    masks: list[Array] = []
    for weights, (prefix_k, prefix_v), trace in zip(layers, prefixes, traces):
        mask = branch_recompute_mask(h, parent, delta)
        masks.append(mask)
        result = specfold_layer(
            h,
            parent,
            mask,
            weights,
            prefix_k=prefix_k,
            prefix_v=prefix_v,
            trace=trace,
        )
        results.append(result)
        h = result.hidden
    return h, results, masks
