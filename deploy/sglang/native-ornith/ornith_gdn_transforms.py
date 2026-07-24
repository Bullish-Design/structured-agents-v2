"""Pure GGUF->HF numeric weight transforms for the Ornith-1.0-9B GDN spike.

The ``qwen35`` GGUF stores certain Gated-DeltaNet weights in a different
*numeric domain* and *head order* than SGLang's native ``Qwen3_5`` model
(which follows HF Transformers). The existing nine spike patches fix tensor
names/binding/config but never touch tensor *values*, so the model loads and
serves but emits gibberish. Three value transforms are missing:

1. **RMSNorm -1** -- GGUF norm weights are ``1 + w``; SGLang's ``GemmaRMSNorm``
   adds 1 internally, so subtract 1. Applies to ``input_layernorm``,
   ``post_attention_layernorm``, final ``model.norm``, and the full-attention
   ``q_norm``/``k_norm`` -- **not** ``ssm_norm`` (RMSNormGated, no offset).
2. **A_log domain** -- GGUF ``ssm_a`` is the materialized ``A = -exp(A_log)``;
   SGLang wants ``A_log = log(-ssm_a)``.
3. **Value-head unpermutation** -- llama.cpp lays the 32 GQA-repeated linear
   value heads in ``(repeat=2, groups=16)`` row-major order; HF/SGLang wants
   ``(groups=16, repeat=2)``.

This module is intentionally free of SGLang imports so the math can be
unit-tested in isolation (see ``test_ornith_gdn_transforms.py``). The two
dispatch functions match on the HF module path and are tolerant of a trailing
``.weight`` / ``.qweight`` / bare suffix. Each of the three transforms can be
toggled off (for incremental bisection against a reference oracle) via keyword
flags; the caller in ``ornith_text_model.load_weights`` reads the env once and
passes them through.

9B GDN constants (verified from the GGUF this session -- see the project
IMPLEMENTATION-GUIDE): groups ``g=16``, repeat ``r=2``, head_dim ``d=128``,
so ``v_heads=32``, ``V = r*g*d = 4096``, and the qkv/conv V-section begins at
``QK = 4096`` (Q 2048 + K 2048).
"""

from __future__ import annotations

import torch

G: int = 16  # groups (== linear k-heads)
R: int = 2  # repeat (v_heads / k_heads)
D: int = 128  # head_dim
QK: int = 4096  # Q(2048)+K(2048); V section starts here in qkv & conv1d
V: int = R * G * D  # 4096 (v_heads * head_dim)


def _perm_head() -> torch.Tensor:
    """Length-32 permutation mapping HF v-head order -> GGUF v-head index.

    GGUF order is ``(r, g)`` row-major (``h_gguf = r*G + g``); HF order is
    ``(g, r)`` (``h_hf = g*R + r``). ``perm[h_hf] = h_gguf`` gathers a
    GGUF-ordered head axis into HF order. Equals ``[0,16,1,17,...,15,31]``.
    """
    p = torch.arange(R).view(R, 1) * G + torch.arange(G).view(1, G)  # (R,G) gguf
    return p.permute(1, 0).reshape(-1)  # (G,R) flat == HF-order -> gguf idx


def _perm_rows(width: int) -> torch.Tensor:
    """Expand ``_perm_head`` to a row-index permutation for width-``width`` blocks.

    For a per-head *row block* of ``width`` rows (e.g. ``in_proj_z``'s 128 rows
    per head), reorder whole blocks into HF head order without splitting them.
    """
    p = _perm_head()
    return (p.view(-1, 1) * width + torch.arange(width).view(1, width)).reshape(-1)


_NORM_SUFFIXES: tuple[str, ...] = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "model.norm.weight",
    "q_norm.weight",
    "k_norm.weight",
)


def transform_plain(
    hf_name: str,
    t: torch.Tensor,
    *,
    fix_norm: bool = True,
    fix_alog: bool = True,
    fix_perm: bool = True,
) -> torch.Tensor:
    """Apply the value/order fix to an UNQUANTIZED (plain ``.weight``) tensor.

    Covers the F32 GDN tensors (norms, ``A_log``, ``dt_bias``, ``in_proj_a``/
    ``in_proj_b``, ``conv1d``) and the unquantized-built ``out_proj`` (column
    permute, guide S4.3 Option A). Unmatched names pass through unchanged.
    """
    n = hf_name
    if n.endswith(_NORM_SUFFIXES):
        if not fix_norm:
            return t
        return (t.float() - 1.0).to(t.dtype)  # RMSNorm -1

    if n.endswith("linear_attn.A_log"):
        x = t
        if fix_alog:
            x = (-x.float()).log().to(t.dtype)  # A = -exp(A_log) -> A_log
        if fix_perm:
            x = x[_perm_head()]
        return x

    if n.endswith("linear_attn.dt_bias"):
        return t[_perm_head()] if fix_perm else t

    if n.endswith(("linear_attn.in_proj_a.weight", "linear_attn.in_proj_b.weight")):
        # [v_heads=32, H] -- one row per head.
        return t[_perm_head()] if fix_perm else t

    if n.endswith("linear_attn.conv1d.weight") and t.shape[0] > QK:
        if not fix_perm:
            return t
        qk, v = t[:QK], t[QK:]
        v = v[_perm_rows(D)]  # V-section rows only
        return torch.cat([qk, v], 0)

    if n.endswith("linear_attn.out_proj.weight"):
        # [H, V] built unquantized (Option A) -- permute input columns at
        # 128-col head granularity: (H,R,G,D).permute(0,2,1,3).
        if not fix_perm:
            return t
        h = t.shape[0]
        return (
            t.reshape(h, R, G, D)
            .permute(0, 2, 1, 3)
            .contiguous()
            .reshape(h, R * G * D)
        )

    return t


def transform_quant_rows(
    hf_name: str,
    qweight: torch.Tensor,
    *,
    fix_perm: bool = True,
) -> torch.Tensor:
    """Row (output-dim) permutation on a packed GGUF ``qweight`` -- SAFE.

    ``qweight`` is ``[out_features, packed_bytes]``; llama.cpp quantizes along
    the input dim (now packed into dim 1), so reordering whole rows never
    splits a quant block (DECISIONS D5). Only the value-head axes are touched:
    all rows of ``in_proj_z``, and the V-section (rows 4096:8192) of
    ``in_proj_qkv``. Unmatched names pass through unchanged.
    """
    if not fix_perm:
        return qweight
    if hf_name.endswith("linear_attn.in_proj_z.qweight"):
        return qweight[_perm_rows(D)]  # all V rows
    if hf_name.endswith("linear_attn.in_proj_qkv.qweight"):
        qk, v = qweight[:QK], qweight[QK:]
        return torch.cat([qk, v[_perm_rows(D)]], 0)
    if hf_name.endswith("linear_attn.out_proj.qweight"):
        return _perm_out_proj_cols_packed(qweight)
    return qweight


def _perm_out_proj_cols_packed(qweight: torch.Tensor) -> torch.Tensor:
    """Head-granular *column* (input-dim) permute on a packed ``out_proj``.

    ``out_proj`` ``[H, V]`` needs its V-head input axis reordered GGUF->HF, but
    that axis is the *input* dim -- packed into dim 1. This can't be row-indexed
    (guide S4.3). It is only valid because this GGUF stores ``ssm_out`` as
    **Q8_0**: 4352 packed bytes/row = 32 heads x 136 bytes, and 136 = 4 whole
    Q8_0 blocks (34 bytes each, 32 weights/block). A 128-input-feature head
    chunk is thus exactly 4 whole blocks, so reordering the 32 per-head 136-byte
    segments never splits a quant block (guide S4.3 Option B). If the packed
    width is not an exact multiple of the head count (a different quant type),
    fall back to leaving it unpermuted rather than corrupt the bytes.
    """
    heads = R * G  # 32 value heads
    out_rows, packed_bytes = qweight.shape
    if packed_bytes % heads != 0:
        return qweight
    seg = packed_bytes // heads
    q = qweight.reshape(out_rows, heads, seg)
    q = q[:, _perm_head(), :]
    return q.contiguous().reshape(out_rows, heads * seg)
