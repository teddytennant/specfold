"""CPU tests for the SpecFold folded layer. float64, seeded, no kernels."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pytest

from specfold import (
    ProjectionTrace,
    SpecFoldLayer,
    SpecFoldWeights,
    apply_rope,
    branch_recompute_mask,
    correct_attention,
    ffn_from_hidden,
    qkv_from_hidden,
    recompute_mask,
    relative_residual,
    rms_norm,
    specfold_stack,
    streaming_attention_state,
    swiglu,
)
from specfold.attention import concat_prefix
from specfold.layer import attention_output_proj, block_positions


D_MODEL = 16
N_HEADS = 2
D_HEAD = 8
D_FF = 32


def _weights(key, d_model=D_MODEL, n_heads=N_HEADS, d_head=D_HEAD, d_ff=D_FF):
    keys = jax.random.split(key, 7)

    def n(k, shape, scale):
        return scale * jax.random.normal(k, shape, dtype=jnp.float64)

    return SpecFoldWeights(
        attn_norm=1.0 + 0.1 * jax.random.normal(keys[0], (d_model,), dtype=jnp.float64),
        ffn_norm=1.0 + 0.1 * jax.random.normal(keys[1], (d_model,), dtype=jnp.float64),
        w_qkv=n(keys[2], (d_model, 3 * n_heads * d_head), 0.25),
        w_o=n(keys[3], (n_heads * d_head, d_model), 0.25),
        w_gate=n(keys[4], (d_model, d_ff), 0.25),
        w_up=n(keys[5], (d_model, d_ff), 0.25),
        w_down=n(keys[6], (d_ff, d_model), 0.25),
        n_heads=n_heads,
        rope_theta=10_000.0,
        rms_eps=1e-6,
    )


def _randn(key, shape):
    return jax.random.normal(key, shape, dtype=jnp.float64)


def _max_abs(a, b):
    return float(jnp.max(jnp.abs(a - b)))


def _full_softmax(q, k, v):
    """Independent of the streaming state and of eq 8."""
    scale = jnp.reciprocal(jnp.sqrt(jnp.asarray(q.shape[-1], dtype=q.dtype)))
    logits = jnp.einsum("qhd,khd->qhk", q, k) * scale
    weights = jax.nn.softmax(logits, axis=-1)
    return jnp.einsum("qhk,khd->qhd", weights, v)


def _ref_order(parent):
    parent = [int(p) for p in parent]
    seen = set()
    order = []

    def visit(i):
        if i in seen:
            return
        if parent[i] >= 0:
            visit(parent[i])
        seen.add(i)
        order.append(i)

    for i in range(len(parent)):
        visit(i)
    return order


def _reference_folded(hidden, parent, recompute, weights, prefix_k=None, prefix_v=None):
    """Folded forward that never uses the incremental correction.

    Inherit QKV at folded positions, recompute QKV at recomputed positions,
    full softmax over prefix + the branch KV, residual from the child's own
    hidden state, FFN only at recomputed positions.
    """
    hidden = jnp.asarray(hidden)
    recompute = jnp.asarray(recompute, dtype=bool)
    branches, n_positions, _ = hidden.shape
    heads, d_head = weights.n_heads, weights.d_head
    dtype = hidden.dtype
    if prefix_k is None:
        prefix_k = jnp.zeros((0, heads, d_head), dtype=dtype)
        prefix_v = jnp.zeros((0, heads, d_head), dtype=dtype)
    positions = block_positions(n_positions, int(prefix_k.shape[0]), dtype)
    parent = [int(p) for p in parent]

    q = [None] * branches
    k = [None] * branches
    v = [None] * branches
    ffn = [None] * branches
    h_plus = [None] * branches
    h_out = [None] * branches

    for i in _ref_order(parent):
        idx = jnp.where(recompute[i])[0]
        p = parent[i]
        if p < 0:
            qi = jnp.zeros((n_positions, heads, d_head), dtype=dtype)
            ki = jnp.zeros_like(qi)
            vi = jnp.zeros_like(qi)
            fi = jnp.zeros((n_positions, hidden.shape[-1]), dtype=dtype)
        else:
            qi, ki, vi, fi = q[p], k[p], v[p], ffn[p]
        if idx.shape[0] != 0:
            qr, kr, vr = qkv_from_hidden(hidden[i, idx], positions[idx], weights)
            qi = qi.at[idx].set(qr)
            ki = ki.at[idx].set(kr)
            vi = vi.at[idx].set(vr)
        o = _full_softmax(qi, concat_prefix(prefix_k, ki), concat_prefix(prefix_v, vi))
        hp = hidden[i] + attention_output_proj(o, weights.w_o)
        if idx.shape[0] != 0:
            fi = fi.at[idx].set(ffn_from_hidden(hp[idx], weights))
        q[i], k[i], v[i], ffn[i] = qi, ki, vi, fi
        h_plus[i] = hp
        h_out[i] = hp + fi

    return (
        jnp.stack(h_out),
        jnp.stack(h_plus),
        jnp.stack(q),
        jnp.stack(k),
        jnp.stack(v),
    )


def _dense_block(h, weights, prefix_k=None, prefix_v=None):
    """Standard pre-norm block, written out rather than calling SpecFoldLayer."""
    n_positions = h.shape[0]
    heads, d_head = weights.n_heads, weights.d_head
    if prefix_k is None:
        prefix_k = jnp.zeros((0, heads, d_head), dtype=h.dtype)
        prefix_v = jnp.zeros((0, heads, d_head), dtype=h.dtype)
    positions = jnp.arange(prefix_k.shape[0], prefix_k.shape[0] + n_positions)
    x = rms_norm(h, weights.attn_norm, weights.rms_eps)
    packed = (x @ weights.w_qkv).reshape(n_positions, 3, heads, d_head)
    q = apply_rope(packed[:, 0], positions, weights.rope_theta)
    k = apply_rope(packed[:, 1], positions, weights.rope_theta)
    v = packed[:, 2]
    o = _full_softmax(q, concat_prefix(prefix_k, k), concat_prefix(prefix_v, v))
    h_plus = h + o.reshape(n_positions, heads * d_head) @ weights.w_o
    y = rms_norm(h_plus, weights.ffn_norm, weights.rms_eps)
    ffn = swiglu(y, weights.w_gate, weights.w_up, weights.w_down)
    return h_plus + ffn


def _correct_without_rescale(q, k_parent, v_parent, k_child, v_child, m, z, o, index):
    """Eq 8 with f omitted (forced to 1). Used only to show the test is sensitive."""
    scale = jnp.reciprocal(jnp.sqrt(jnp.asarray(q.shape[-1], dtype=q.dtype)))
    k_old, v_old = k_parent[index], v_parent[index]
    k_new, v_new = k_child[index], v_child[index]
    s_old = jnp.einsum("fhd,rhd->fhr", q, k_old) * scale
    s_new = jnp.einsum("fhd,rhd->fhr", q, k_new) * scale
    m_prime = jnp.maximum(m, jnp.max(s_new, axis=-1))
    a_old = jnp.exp(s_old - m_prime[..., None])
    a_new = jnp.exp(s_new - m_prime[..., None])
    z_prime = z - jnp.sum(a_old, axis=-1) + jnp.sum(a_new, axis=-1)
    acc = o * z[..., None]
    acc = acc - jnp.einsum("fhr,rhd->fhd", a_old, v_old)
    acc = acc + jnp.einsum("fhr,rhd->fhd", a_new, v_new)
    return acc / z_prime[..., None]


def _correct_without_subtraction(q, k_parent, v_parent, k_child, v_child, m, z, o, index):
    """Eq 8 without removing the parent's contributions at R_i."""
    del k_parent, v_parent
    scale = jnp.reciprocal(jnp.sqrt(jnp.asarray(q.shape[-1], dtype=q.dtype)))
    k_new, v_new = k_child[index], v_child[index]
    s_new = jnp.einsum("fhd,rhd->fhr", q, k_new) * scale
    m_prime = jnp.maximum(m, jnp.max(s_new, axis=-1))
    rescale = jnp.exp(m - m_prime)
    a_new = jnp.exp(s_new - m_prime[..., None])
    z_prime = z * rescale + jnp.sum(a_new, axis=-1)
    acc = o * z[..., None] * rescale[..., None]
    acc = acc + jnp.einsum("fhr,rhd->fhd", a_new, v_new)
    return acc / z_prime[..., None]


def test_gate_threshold_main_and_zero_parent():
    # Parent L2 norm is 2**20. The threshold step is one ulp of that scale,
    # so "just below" and "just above" are exact in float64.
    delta = 0.5
    parent = jnp.zeros(D_MODEL, dtype=jnp.float64).at[0].set(2.0**20)
    just_below = parent.at[0].set(2.0**20 + (2.0**19 - 1))
    equal = parent.at[0].set(2.0**20 + 2.0**19)
    just_above = parent.at[0].set(2.0**20 + (2.0**19 + 1))

    rho_below = relative_residual(just_below, parent)
    rho_equal = relative_residual(equal, parent)
    rho_above = relative_residual(just_above, parent)
    assert float(rho_below) == pytest.approx(0.5 - 2.0**-20)
    assert float(rho_equal) == 0.5
    assert float(rho_above) == pytest.approx(0.5 + 2.0**-20)
    assert rho_below < delta
    assert rho_above > delta
    assert not bool(recompute_mask(just_below, parent, delta))
    assert not bool(recompute_mask(equal, parent, delta))
    assert bool(recompute_mask(just_above, parent, delta))

    # L2, not L1. parent = (3, 4) has L2 norm 5 and L1 norm 7.
    parent_l2 = jnp.zeros(D_MODEL, dtype=jnp.float64).at[0].set(3.0).at[1].set(4.0)
    diff = jnp.zeros(D_MODEL, dtype=jnp.float64).at[1].set(5.0)
    rho = relative_residual(parent_l2 + diff, parent_l2)
    assert rho == pytest.approx(1.0)
    assert abs(float(rho) - (5.0 / 7.0)) > 0.1

    # Child positions: identical, just below, just above. Main branch holds the parent.
    hidden = jnp.stack(
        [
            jnp.stack([parent, parent, parent]),
            jnp.stack([parent, just_below, just_above]),
        ]
    )
    mask = branch_recompute_mask(hidden, [-1, 0], delta)
    assert bool(jnp.all(mask[0]))
    assert not bool(mask[1, 0])
    assert not bool(mask[1, 1])
    assert bool(mask[1, 2])

    # Main branch recomputes even if a parent index is set and rho would fold.
    forced = jnp.stack([just_below, parent])[:, None, :]
    mask_main = branch_recompute_mask(forced, [1, -1], delta)
    assert bool(mask_main[0, 0])
    assert not bool(recompute_mask(just_below, parent, delta))

    zero = jnp.zeros(D_MODEL, dtype=jnp.float64)
    rho_zero = relative_residual(jnp.ones(D_MODEL, dtype=jnp.float64), zero)
    rho_both = relative_residual(zero, zero)
    assert not bool(jnp.isnan(rho_zero))
    assert not bool(jnp.isnan(rho_both))
    assert bool(jnp.isinf(rho_zero))
    assert bool(jnp.isinf(rho_both))
    assert bool(recompute_mask(jnp.ones(D_MODEL, dtype=jnp.float64), zero, delta))
    assert bool(recompute_mask(zero, zero, delta))

    batch = jnp.stack([zero, jnp.ones(D_MODEL, dtype=jnp.float64)])
    rho_batch = relative_residual(batch, jnp.stack([zero, zero]))
    assert not bool(jnp.any(jnp.isnan(rho_batch)))


def test_attention_correction_matches_full_softmax():
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, 8)
    n_pos, prefix_len, heads, d_head = 6, 3, N_HEADS, D_HEAD
    q = _randn(keys[0], (n_pos, heads, d_head))
    k = _randn(keys[1], (n_pos, heads, d_head))
    v = _randn(keys[2], (n_pos, heads, d_head))
    prefix_k = _randn(keys[3], (prefix_len, heads, d_head))
    prefix_v = _randn(keys[4], (prefix_len, heads, d_head))
    index = jnp.array([1, 4])

    m, z, o = streaming_attention_state(
        q, concat_prefix(prefix_k, k), concat_prefix(prefix_v, v)
    )
    o_parent = _full_softmax(q, concat_prefix(prefix_k, k), concat_prefix(prefix_v, v))
    assert _max_abs(o, o_parent) < 1e-12

    # New keys at R are large and positive so some scores exceed the old max
    # (f is not 1) and old contributions differ from new ones.
    k_child = k.at[index].set(jnp.abs(k[index]) + 4.0)
    v_child = v.at[index].set(_randn(keys[5], (index.shape[0], heads, d_head)) * 3.0)
    m2, z2, o2 = correct_attention(q, k, v, k_child, v_child, m, z, o, index)
    o_ref = _full_softmax(
        q, concat_prefix(prefix_k, k_child), concat_prefix(prefix_v, v_child)
    )
    err_random = _max_abs(o2, o_ref)
    assert err_random < 1e-6
    assert bool(jnp.any(m2 > m + 1e-6))

    broken_f = _correct_without_rescale(q, k, v, k_child, v_child, m, z, o, index)
    broken_sub = _correct_without_subtraction(
        q, k, v, k_child, v_child, m, z, o, index
    )
    assert _max_abs(broken_f, o_ref) > 1e-2
    assert _max_abs(broken_sub, o_ref) > 1e-2

    # Parent max sits on a replaced key and the child's score there drops, so
    # the paper's m' stays above the true max. O must still match.
    q_nt = jnp.zeros((2, 1, 2), dtype=jnp.float64).at[:, :, 0].set(1.0)
    k_nt = jnp.zeros((3, 1, 2), dtype=jnp.float64)
    k_nt = k_nt.at[1, :, 0].set(5.0).at[2, :, 0].set(0.5)
    v_nt = jnp.zeros((3, 1, 2), dtype=jnp.float64)
    v_nt = v_nt.at[1, :, 0].set(3.0).at[2, :, 1].set(1.0)
    prefix_k_nt = jnp.zeros((1, 1, 2), dtype=jnp.float64)
    prefix_v_nt = jnp.ones((1, 1, 2), dtype=jnp.float64)
    m_nt, z_nt, o_nt = streaming_attention_state(
        q_nt, concat_prefix(prefix_k_nt, k_nt), concat_prefix(prefix_v_nt, v_nt)
    )
    k_nt_child = k_nt.at[1].set(0.0)
    v_nt_child = v_nt.at[1].set(0.0)
    m_c, z_c, o_c = correct_attention(
        q_nt, k_nt, v_nt, k_nt_child, v_nt_child, m_nt, z_nt, o_nt, jnp.array([1])
    )
    o_nt_ref = _full_softmax(
        q_nt,
        concat_prefix(prefix_k_nt, k_nt_child),
        concat_prefix(prefix_v_nt, v_nt_child),
    )
    err_nontight = _max_abs(o_c, o_nt_ref)
    assert err_nontight < 1e-6
    true_max = jnp.max(
        jnp.einsum(
            "qhd,khd->qhk",
            q_nt,
            concat_prefix(prefix_k_nt, k_nt_child),
        )
        / jnp.sqrt(2.0),
        axis=-1,
    )
    assert bool(jnp.all(m_c > true_max + 1.0))

    # Folded-query subset, the way the layer calls the correction.
    folded = jnp.array([0, 2, 3, 5])
    m_f, z_f, o_f = correct_attention(
        q[folded], k, v, k_child, v_child, m[folded], z[folded], o[folded], index
    )
    err_folded = _max_abs(o_f, o_ref[folded])
    assert err_folded < 1e-6


def test_layer_matches_reference_folded_forward_with_chain():
    key = jax.random.PRNGKey(1)
    keys = jax.random.split(key, 6)
    n_pos, prefix_len, branches = 6, 2, 4
    weights = _weights(keys[0])
    hidden = _randn(keys[1], (branches, n_pos, D_MODEL))
    prefix_k = _randn(keys[2], (prefix_len, N_HEADS, D_HEAD))
    prefix_v = _randn(keys[3], (prefix_len, N_HEADS, D_HEAD))
    # Chain 0 -> 1 -> 2 -> 3. Position 0 is folded by 1, 2, and 3 (depth 3).
    # Position 1 is folded by the middle branch (1) and recomputed by its child.
    # Position 2 is folded by branch 2 and recomputed by branch 3.
    recompute = jnp.array(
        [
            [True, True, True, True, True, True],
            [False, False, True, True, True, True],
            [False, True, False, True, True, False],
            [False, True, True, False, True, True],
        ]
    )
    parent = [-1, 0, 1, 2]
    layer = SpecFoldLayer(weights)
    result = layer(hidden, parent, recompute, prefix_k, prefix_v)
    ref_h, ref_hp, ref_q, ref_k, ref_v = _reference_folded(
        hidden, parent, recompute, weights, prefix_k, prefix_v
    )
    err_h = _max_abs(result.hidden, ref_h)
    err_hp = _max_abs(result.h_plus, ref_hp)
    err_q = _max_abs(result.q, ref_q)
    err_kv = max(_max_abs(result.k, ref_k), _max_abs(result.v, ref_v))
    assert err_h < 1e-8
    assert err_hp < 1e-8
    assert err_q < 1e-12
    assert err_kv < 1e-12
    # The deepest folded position is not a copy of the main branch's attention
    # output: intermediate recomputes change the KV the correction must track.
    assert _max_abs(result.o[3, 0], result.o[0, 0]) > 1e-3


def test_delta_zero_matches_dense_prenorm():
    key = jax.random.PRNGKey(2)
    keys = jax.random.split(key, 5)
    n_pos, branches = 5, 3
    weights = _weights(keys[0])
    hidden = _randn(keys[1], (branches, n_pos, D_MODEL))
    # Separate each branch so rho > 0 everywhere and delta = 0 folds nothing.
    hidden = hidden + jnp.arange(branches, dtype=jnp.float64)[:, None, None]
    parent = [-1, 0, 1]
    mask = branch_recompute_mask(hidden, parent, delta=0.0)
    assert bool(jnp.all(mask))

    prefix_k = _randn(keys[2], (2, N_HEADS, D_HEAD))
    prefix_v = _randn(keys[3], (2, N_HEADS, D_HEAD))
    result = SpecFoldLayer(weights)(hidden, parent, mask, prefix_k, prefix_v)
    for i in range(branches):
        dense = _dense_block(hidden[i], weights, prefix_k, prefix_v)
        assert _max_abs(result.hidden[i], dense) < 1e-8

    # P = 0, explicit all-recompute mask, no dependence on the gate.
    result0 = SpecFoldLayer(weights)(
        hidden, parent, jnp.ones((branches, n_pos), dtype=bool)
    )
    for i in range(branches):
        dense = _dense_block(hidden[i], weights)
        assert _max_abs(result0.hidden[i], dense) < 1e-8


def test_projections_only_on_recomputed_positions():
    key = jax.random.PRNGKey(3)
    keys = jax.random.split(key, 3)
    n_pos, branches = 5, 3
    weights = _weights(keys[0])
    hidden = _randn(keys[1], (branches, n_pos, D_MODEL))
    recompute = jnp.array(
        [
            [True, True, True, True, True],
            [False, True, False, True, False],
            [False, False, True, False, True],
        ]
    )
    trace = ProjectionTrace()
    SpecFoldLayer(weights, trace=trace)(hidden, [-1, 0, 1], recompute)
    qkv = set(trace.qkv_rows)
    ffn = set(trace.ffn_rows)
    for b in range(branches):
        for n in range(n_pos):
            pair = (b, n)
            if bool(recompute[b, n]):
                assert trace.qkv_rows.count(pair) == 1
                assert trace.ffn_rows.count(pair) == 1
            else:
                assert pair not in qkv
                assert pair not in ffn
    # A fully folded branch never touches the projections.
    assert all(b != 1 or n in (1, 3) for b, n in qkv)


def test_residual_uses_child_hidden_at_folded_positions():
    key = jax.random.PRNGKey(4)
    keys = jax.random.split(key, 3)
    n_pos = 4
    weights = _weights(keys[0])
    base = _randn(keys[1], (n_pos, D_MODEL))
    hidden = jnp.stack([base, base, base])
    hidden = hidden.at[1, 0].set(base[0] + jnp.array([1.5, -0.5] + [0.0] * (D_MODEL - 2)))
    hidden = hidden.at[2, 0].set(base[0] + jnp.array([-2.0, 0.75] + [0.0] * (D_MODEL - 2)))
    recompute = jnp.array(
        [
            [True, True, True, True],
            [False, False, False, False],
            [False, False, False, False],
        ]
    )
    result = SpecFoldLayer(weights)(hidden, [-1, 0, 0], recompute)
    # Inherited Q, K, V match the parent at the folded position.
    for child in (1, 2):
        assert _max_abs(result.q[child, 0], result.q[0, 0]) == 0.0
        assert _max_abs(result.k[child, 0], result.k[0, 0]) == 0.0
        assert _max_abs(result.v[child, 0], result.v[0, 0]) == 0.0
        assert _max_abs(result.q[child], result.q[0]) == 0.0
    # Residual add uses the child's own H, so H_plus differs by exactly that.
    assert _max_abs(result.h_plus[1, 0], result.h_plus[2, 0]) > 1.0
    assert _max_abs(
        result.h_plus[1, 0] - result.h_plus[2, 0],
        hidden[1, 0] - hidden[2, 0],
    ) < 1e-12
    assert _max_abs(
        result.h_plus[1, 0] - result.h_plus[0, 0],
        hidden[1, 0] - hidden[0, 0],
    ) < 1e-12


def test_two_layer_stack_matches_reference():
    key = jax.random.PRNGKey(5)
    keys = jax.random.split(key, 4)
    n_pos, branches = 5, 3
    layers = [_weights(keys[0]), _weights(keys[1])]
    base = _randn(keys[2], (n_pos, D_MODEL))
    hidden = jnp.stack([base, base, base])
    # Branch 1 diverges at position 2; branch 2 diverges from branch 1 at 2 and 4.
    hidden = hidden.at[1, 2].add(3.0)
    hidden = hidden.at[2, 4].add(-2.5)
    parent = [-1, 0, 1]
    delta = 0.05

    final, results, masks = specfold_stack(hidden, parent, layers, delta)
    assert bool(jnp.all(jnp.isfinite(final)))
    # Layer 0 must actually fold something and recompute something.
    assert bool(jnp.any(~masks[0][1:]))
    assert bool(jnp.any(masks[0][1:]))

    h = hidden
    for weights in layers:
        mask = branch_recompute_mask(h, parent, delta)
        h, _, _, _, _ = _reference_folded(h, parent, mask, weights)
    assert bool(jnp.all(jnp.isfinite(h)))
    assert _max_abs(final, h) < 1e-8
    # The stack's first layer agrees with one reference application, not only the end.
    ref0, _, _, _, _ = _reference_folded(hidden, parent, masks[0], layers[0])
    assert _max_abs(results[0].hidden, ref0) < 1e-8
