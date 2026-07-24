"""Unit tests for the pure GGUF->HF GDN weight transforms (9B constants).

Runnable both under pytest and as a plain script (``python
test_ornith_gdn_transforms.py``) since the spike's devenv has torch (via
sglang) but not necessarily pytest. Every check is a bare ``assert`` so the
script form exits non-zero on failure.
"""

from __future__ import annotations

import torch

import ornith_gdn_transforms as T


def test_perm_head_is_expected_bijection() -> None:
    perm = T._perm_head()
    # HF-order -> gguf-index interleave: [0,16,1,17,...,15,31].
    expected = torch.tensor(
        [v for g in range(16) for v in (g, g + 16)], dtype=perm.dtype
    )
    assert torch.equal(perm, expected)
    # Bijection of range(32).
    assert torch.equal(torch.sort(perm).values, torch.arange(32))
    # It is not the identity (a real reorder happens).
    assert not torch.equal(perm, torch.arange(32))


def test_perm_rows_matches_reference_tensor_op_scalar() -> None:
    # [32]-shaped per-head scalar: index gather == reshape(R,G).permute(1,0).
    x = torch.randn(32, dtype=torch.float64)
    ref = x.reshape(T.R, T.G).permute(1, 0).contiguous().reshape(-1)
    got = x[T._perm_head()]
    assert torch.equal(ref, got)


def test_perm_rows_matches_reference_tensor_op_rowblock() -> None:
    # [V, H] row-block: index gather via _perm_rows(D) ==
    # reshape(R,G,D,H).permute(1,0,2,3).
    H = 7
    V = T.R * T.G * T.D
    x = torch.randn(V, H, dtype=torch.float64)
    ref = x.reshape(T.R, T.G, T.D, H).permute(1, 0, 2, 3).contiguous().reshape(V, H)
    got = x[T._perm_rows(T.D)]
    assert torch.equal(ref, got)


def test_log_neg_inverts_neg_exp() -> None:
    y = torch.randn(32, dtype=torch.float64)
    ssm_a = -torch.exp(y)  # GGUF stores A = -exp(A_log)
    recovered = (-ssm_a).log()
    assert torch.allclose(recovered, y, atol=1e-10)


def test_transform_plain_norm_subtracts_one() -> None:
    w = torch.randn(16)
    out = T.transform_plain("model.layers.0.input_layernorm.weight", w)
    assert torch.allclose(out, w - 1.0)
    # ssm_norm must NOT be touched (GroupNorm/RMSNormGated, no offset).
    ssm = torch.randn(16)
    out2 = T.transform_plain("model.layers.0.linear_attn.norm.weight", ssm)
    assert torch.equal(out2, ssm)


def test_transform_plain_alog_domain_and_perm() -> None:
    y = torch.randn(32, dtype=torch.float32)
    ssm_a = -torch.exp(y)
    out = T.transform_plain("model.layers.0.linear_attn.A_log", ssm_a)
    expected = y[T._perm_head()]
    assert torch.allclose(out, expected, atol=1e-5)


def test_transform_plain_dt_bias_perm_only() -> None:
    b = torch.randn(32)
    out = T.transform_plain("model.layers.0.linear_attn.dt_bias", b)
    assert torch.equal(out, b[T._perm_head()])


def test_transform_plain_ba_rowperm() -> None:
    H = 4096
    w = torch.randn(32, H)
    out = T.transform_plain("model.layers.0.linear_attn.in_proj_a.weight", w)
    assert torch.equal(out, w[T._perm_head()])


def test_transform_plain_conv1d_v_section_only() -> None:
    # conv1d arrives 3-D [8192, 1, 4]; only the V section (rows 4096:) permutes.
    chans = T.QK + T.V
    w = torch.randn(chans, 1, 4)
    out = T.transform_plain("model.layers.0.linear_attn.conv1d.weight", w)
    assert torch.equal(out[: T.QK], w[: T.QK])  # Q,K rows untouched
    ref_v = (
        w[T.QK :]
        .reshape(T.R, T.G, T.D, 1, 4)
        .permute(1, 0, 2, 3, 4)
        .contiguous()
        .reshape(T.V, 1, 4)
    )
    assert torch.equal(out[T.QK :], ref_v)


def test_transform_plain_out_proj_column_perm() -> None:
    H = 4096
    o = torch.randn(H, T.V)
    out = T.transform_plain("model.layers.0.linear_attn.out_proj.weight", o)
    ref = (
        o.reshape(H, T.R, T.G, T.D)
        .permute(0, 2, 1, 3)
        .contiguous()
        .reshape(H, T.V)
    )
    assert torch.equal(out, ref)


def test_transform_quant_rows_qkv_v_section_only() -> None:
    # Packed qweight [out, packed_bytes]; only rows 4096:8192 reorder.
    packed = 3360
    q = torch.randint(0, 256, (T.QK + T.V, packed), dtype=torch.uint8)
    out = T.transform_quant_rows("model.layers.0.linear_attn.in_proj_qkv.qweight", q)
    assert torch.equal(out[: T.QK], q[: T.QK])
    assert torch.equal(out[T.QK :], q[T.QK :][T._perm_rows(T.D)])


def test_transform_quant_rows_z_all_rows() -> None:
    packed = 2816
    q = torch.randint(0, 256, (T.V, packed), dtype=torch.uint8)
    out = T.transform_quant_rows("model.layers.0.linear_attn.in_proj_z.qweight", q)
    assert torch.equal(out, q[T._perm_rows(T.D)])


def test_transform_quant_rows_out_proj_packed_cols() -> None:
    # Packed Q8_0 out_proj: [H, 32*136]; reorder per-head 136-byte segments by
    # perm_head, keeping each segment's bytes intact.
    H, heads, seg = 5, T.R * T.G, 136
    # Give each (row, head) segment a distinct constant so we can track it.
    q = torch.zeros(H, heads * seg, dtype=torch.uint8)
    for h in range(heads):
        q[:, h * seg : (h + 1) * seg] = h % 256
    out = T.transform_quant_rows(
        "model.layers.0.linear_attn.out_proj.qweight", q
    )
    perm = T._perm_head()
    for new_h in range(heads):
        src_h = int(perm[new_h])
        block = out[:, new_h * seg : (new_h + 1) * seg]
        assert torch.equal(block, torch.full_like(block, src_h % 256))


def test_out_proj_packed_matches_plain_head_reorder() -> None:
    # With seg==D, the packed segment reorder must reproduce the plain
    # column-permute's head ordering (bytes stand in for the D feature cols).
    H = 3
    seg = T.D
    heads = T.R * T.G
    plain = torch.randn(H, T.V, dtype=torch.float64)
    plain_out = T.transform_plain(
        "model.layers.0.linear_attn.out_proj.weight", plain
    )
    # Emulate the packed path on the same head-block layout.
    q = plain.reshape(H, heads, seg)
    packed_out = q[:, T._perm_head(), :].reshape(H, T.V)
    assert torch.equal(plain_out, packed_out)


def test_transform_quant_rows_ignores_unrelated() -> None:
    q = torch.randint(0, 256, (100, 10), dtype=torch.uint8)
    out = T.transform_quant_rows("model.layers.0.mlp.gate_up_proj.qweight", q)
    assert out is q


def test_toggles_disable_each_stage() -> None:
    # fix_perm off => A_log domain-converted but not permuted.
    y = torch.randn(32, dtype=torch.float32)
    ssm_a = -torch.exp(y)
    out = T.transform_plain(
        "model.layers.0.linear_attn.A_log", ssm_a, fix_perm=False
    )
    assert torch.allclose(out, y, atol=1e-5)
    # fix_alog off => permuted raw values, no log domain change.
    out2 = T.transform_plain(
        "model.layers.0.linear_attn.A_log", ssm_a, fix_alog=False
    )
    assert torch.equal(out2, ssm_a[T._perm_head()])
    # fix_norm off => norm passes through unchanged.
    w = torch.randn(16)
    out3 = T.transform_plain(
        "model.layers.0.input_layernorm.weight", w, fix_norm=False
    )
    assert torch.equal(out3, w)
    # fix_perm off => quant rows untouched.
    q = torch.randint(0, 256, (T.V, 8), dtype=torch.uint8)
    out4 = T.transform_quant_rows(
        "model.layers.0.linear_attn.in_proj_z.qweight", q, fix_perm=False
    )
    assert torch.equal(out4, q)


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
