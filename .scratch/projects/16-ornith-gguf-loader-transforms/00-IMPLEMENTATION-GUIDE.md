# Path 1 — Fix Ornith-1.0-9B GGUF gibberish by porting 3 weight transforms into the SGLang loader

**Goal:** make the existing `deploy/sglang/native-ornith/` spike produce *coherent*
text from the `Ornith-1.0-9B-UD-Q4_K_XL.gguf` (Q4-quantized, fits a 12 GB 3060) by
applying, at weight-load time, the three GGUF→HF numerical transforms that the model
requires and that the current 9 patches never perform. Keep the Q4 file (do **not** fall
back to 19 GB bf16), so the VRAM win is preserved.

**Status of the diagnosis:** the spike currently *loads all 427 tensors and serves*, but
output is gibberish. Root cause established in `deploy/sglang/native-ornith/
RESEARCH-FINDINGS.md` §4b and cross-checked against a working reference converter
(`.scratch/vendor/gguf-to-nvfp4/scripts/step1_convert.py`, function
`gguf_tensor_to_torch()`). That converter's README states verbatim that getting these
wrong yields a model that *"loads cleanly but generates garbage"* — our exact symptom.

> **NO SUBAGENTS.** Do every step directly (read/edit/run). See PLAN.md.

---

## 0. TL;DR of the fix

The GGUF (`qwen35`, llama.cpp naming) stores certain weights in a different *numeric
domain* and *head order* than SGLang's native `Qwen3_5` model (which follows HF
Transformers). The current patches fix tensor *names/binding/config* but never touch
tensor *values*. Three transforms are missing:

1. **RMSNorm `+1` offset** — GGUF norm weights are `1 + w`; SGLang wants `w`. Subtract 1.
2. **A_log domain** — GGUF `ssm_a` is the materialized `A = -exp(A_log)`; SGLang wants
   `A_log`. Convert `A_log = log(-ssm_a)`.
3. **Value-head unpermutation** — GGUF stores the GQA-repeated linear-attn value heads in
   `(repeat, groups)` order; SGLang wants `(groups, repeat)`. For the **9B**:
   `repeat = 2`, `groups = 16`, `v_heads = 32`, `head_dim = 128`.

All three are applied in **one place**: `OrnithTextForCausalLM.load_weights`
(`deploy/sglang/native-ornith/ornith_text_model.py`, lines 172–262), by transforming
`loaded_weight` before it reaches each param's `weight_loader`. Two of the affected
tensors are block-quantized and need care (§4).

---

## 1. Ground-truth model dimensions (verified this session)

Read directly from the GGUF (`gguf.GGUFReader`) and corroborated by the Qwen3.5/Qwen3-Next
architecture. **Do not trust the spike README's numbers over these.**

| Quantity | Value | Source |
|---|---|---|
| hidden size `H` | 4096 | `qwen35.embedding_length` |
| GDN linear **k-heads** | 16 | `qwen35.ssm.group_count` = 16 |
| GDN linear **v-heads** | 32 | `ssm_a` shape `[32]`, `ssm_alpha/beta` `[4096,32]` |
| GDN **repeat** `r` = v/k | **2** | 32 / 16 |
| GDN **groups** `g` | **16** | = k-heads |
| GDN **head_dim** `d` | 128 | `qwen35.ssm.state_size` = 128 |
| Q size | 2048 | k_heads·d = 16·128 |
| K size | 2048 | k_heads·d |
| V size | 4096 | v_heads·d = 32·128 |
| qkv (attn_qkv) | 8192 | Q+K+V; matches GGUF `attn_qkv [4096,8192]` |
| Z / gate (attn_gate) | 4096 | v_heads·d; matches GGUF `attn_gate [4096,4096]` |
| conv1d channels | 8192 | over qkv; GGUF `ssm_conv1d [4,8192]`, kernel 4 |
| block_count | 32 | `qwen35.block_count`; layers 3,7,11,… full-attn (interval 4) |

**Full-attention layers** (every 4th, `full_attention_interval=4`, first at index 3) use
head_dim **256** with `partial_rotary_factor=0.25` and `mrope_section=[11,11,10]` — a
*separate* potential issue (see §7, not this fix's target).

> The reference converter is written for the **27B** (`repeat=3`, `v_heads=48`,
> `(3,16)`). **You must use the 9B constants above** (`repeat=2`, `v_heads=32`,
> `(2,16)`). Do not copy the 27B literals.

---

## 2. Per-tensor transform table (9B)

GGUF name → HF/SGLang param name (as remapped by the existing patches) → dtype in this
GGUF → required transform. "Row" = output dim (dim 0); "Col" = input dim.

| GGUF tensor | HF param (incoming `name`) | dtype | Transform |
|---|---|---|---|
| `blk.N.attn_norm.weight` | `…input_layernorm.weight` | F32 | **−1.0** |
| `blk.N.post_attention_norm.weight` | `…post_attention_layernorm.weight` | F32 | **−1.0** |
| `output_norm.weight` | `model.norm.weight` | F32 | **−1.0** |
| `blk.N.attn_q_norm.weight` (full-attn) | `…self_attn.q_norm.weight` | F32 | **−1.0** |
| `blk.N.attn_k_norm.weight` (full-attn) | `…self_attn.k_norm.weight` | F32 | **−1.0** |
| `blk.N.ssm_norm.weight` | `…linear_attn.norm.weight` | F32 | **NONE** (GroupNorm, not RMSNorm) |
| `blk.N.ssm_a` | `…linear_attn.A_log` | F32 `[32]` | **`log(−x)`** then **head-perm(32)** |
| `blk.N.ssm_dt.bias` | `…linear_attn.dt_bias` | F32 `[32]` | **head-perm(32)** |
| `blk.N.ssm_beta.weight` | `…linear_attn.in_proj_b.weight` | F32 `[32,H]` | **head-row-perm** `(2,16,H)->(16,2,H)` |
| `blk.N.ssm_alpha.weight` | `…linear_attn.in_proj_a.weight` | F32 `[32,H]` | **head-row-perm** same |
| `blk.N.attn_qkv.weight` | `…linear_attn.in_proj_qkv.weight` | **Q6_K** `[8192,·]` | **V-section rows only** (rows 4096:8192) head-row-perm; Q,K rows 0:4096 untouched |
| `blk.N.attn_gate.weight` | `…linear_attn.in_proj_z.weight` | **Q5_K** `[4096,·]` | **all rows** head-row-perm `(2,16,128)->(16,2,128)` |
| `blk.N.ssm_out.weight` | `…linear_attn.out_proj.weight` | **Q8_0** `[H,4096]` | **columns** head-perm on input dim (the awkward one, §4.3) |
| `blk.N.ssm_conv1d.weight` | `…linear_attn.conv1d.weight` | F32 `[8192,1,4]` | **V-section rows only** (rows 4096:8192) head-row-perm |
| `token_embd.weight` | `model.embed_tokens.weight` | Q4_K | NONE |
| `output.weight` | `lm_head.weight` | Q6_K | NONE |
| `blk.N.attn_q/k/v/output`, `ffn_*` | standard | Q4_K/Q6_K | NONE |

### The head permutation, defined once
llama.cpp lays 32 v-heads as `(r=2, g=16)`, row-major → head index `h_gguf = ri*16 + gi`.
HF wants `(g=16, r=2)` → `h_hf = gi*2 + ri`. So the permutation that reorders a
v-head-indexed axis **from GGUF order to HF order** is:

```python
# perm_head[h_hf] = source index in gguf order
import torch
G, R = 16, 2                       # 9B: groups=16, repeat=2  (v_heads = 32)
perm_head = (torch.arange(R).view(R, 1) * G + torch.arange(G).view(1, G))  # (2,16) gguf order
perm_head = perm_head.permute(1, 0).reshape(-1)   # (16,2)->flat  == HF order -> gguf idx
# perm_head == tensor([0,16, 1,17, 2,18, ... 15,31])
```

For a **per-head scalar** (`A_log`, `dt_bias`, shape `[32]`): `t = t[perm_head]`.
For a **per-head row block** of width `d=128` (`in_proj_z`, `attn_qkv` V-section,
`in_proj_a/b` where the "row unit" is 1 not 128 — see note): expand to row indices:

```python
d = 128
row_perm = (perm_head.view(-1, 1) * d + torch.arange(d).view(1, d)).reshape(-1)  # len 4096
t = t[row_perm]                    # reorders the 4096 rows into HF head order
```

> **`in_proj_a`/`in_proj_b` note:** these are `[v_heads, H] = [32, H]` — one *row per
> head*, not per (head,dim). So they use `t = t[perm_head]` (the length-32 form), matching
> the converter's `reshape(2,16,D).permute(1,0,2)`.

Equivalent, and clearer for **plain (unquantized) tensors**, is the converter's tensor-op
form — prefer this for F32 tensors:

```python
# in_proj_z / attn_qkv V-section  [4096, H]  ->  (2,16,128,H).permute(1,0,2,3)
V = V.reshape(R, G, d, H).permute(1, 0, 2, 3).contiguous().reshape(R*G*d, H)
# A_log / dt_bias  [32] -> (2,16).permute(1,0)
x = x.reshape(R, G).permute(1, 0).contiguous().reshape(R*G)
# out_proj  [H, 4096] -> (H,2,16,128).permute(0,2,1,3)
o = o.reshape(H, R, G, d).permute(0, 2, 1, 3).contiguous().reshape(H, R*G*d)
# conv1d V-section [4096,1,4] -> (2,16,128,1,4).permute(1,0,2,3,4)
cv = cv.reshape(R, G, d, 1, K).permute(1, 0, 2, 3, 4).contiguous().reshape(R*G*d, 1, K)
```

---

## 3. Where to hook — `OrnithTextForCausalLM.load_weights`

All incoming weights already carry HF names (the existing patches remap GGUF→HF names).
The loop at `ornith_text_model.py:198–261` handles each `(name, loaded_weight)`. Insert a
single transform dispatch **right after the `conv1d` unsqueeze block (line ~216) and
before the tie-embedding / stacked-mapping logic**, so it applies whether the tensor lands
via the stacked mapping (`in_proj_*`) or the default else-branch (`A_log`, `dt_bias`,
norms, `out_proj`, `conv1d`).

Recommended structure — a small pure helper module `ornith_gdn_transforms.py` (new file
in `deploy/sglang/native-ornith/`), unit-tested in isolation (§6), imported by
`load_weights`:

```python
# ornith_gdn_transforms.py  (pure tensor logic — no SGLang imports, so it is unit-testable)
import torch

G, R, D = 16, 2, 128          # 9B GDN: groups, repeat, head_dim
QK = 4096                     # Q(2048)+K(2048); V section starts here in qkv & conv1d
V  = R * G * D                # 4096

def _perm_head():
    p = (torch.arange(R).view(R,1)*G + torch.arange(G).view(1,G)).permute(1,0).reshape(-1)
    return p                  # len 32, HF-order -> gguf index

def _perm_rows(width):
    p = _perm_head()
    return (p.view(-1,1)*width + torch.arange(width).view(1,width)).reshape(-1)

def transform_plain(hf_name: str, t: torch.Tensor) -> torch.Tensor:
    """Apply the value/order fix to an UNQUANTIZED (plain .weight/.bias) GGUF tensor."""
    n = hf_name
    if n.endswith(("input_layernorm.weight", "post_attention_layernorm.weight",
                   "model.norm.weight", "q_norm.weight", "k_norm.weight")):
        return (t.float() - 1.0).to(t.dtype)                      # RMSNorm -1
    if n.endswith("linear_attn.A_log"):
        return (-t.float()).log().to(t.dtype)[_perm_head()]        # domain + head perm
    if n.endswith("linear_attn.dt_bias"):
        return t[_perm_head()]                                     # head perm
    if n.endswith(("linear_attn.in_proj_a.weight", "linear_attn.in_proj_b.weight")):
        return t[_perm_head()]                                     # [32,H] row perm
    if n.endswith("linear_attn.conv1d.weight") and t.shape[0] > QK:
        qk, v = t[:QK], t[QK:]
        v = v[_perm_rows(D)]                                       # V-section rows
        return torch.cat([qk, v], 0)
    return t

def transform_quant_rows(hf_name: str, qweight: torch.Tensor) -> torch.Tensor:
    """Row (output-dim) permutation on a packed GGUF qweight — SAFE for block quant."""
    if hf_name.endswith("linear_attn.in_proj_z.weight"):
        return qweight[_perm_rows(D)]                              # all 4096 rows
    if hf_name.endswith("linear_attn.in_proj_qkv.weight"):
        qk, v = qweight[:QK], qweight[QK:]                         # rows, not features
        return torch.cat([qk, v[_perm_rows(D)]], 0)
    return qweight
```

- **Norms, A_log, dt_bias, conv1d** are F32 → `transform_plain` on `loaded_weight`,
  applied in the else-branch path (they are not in `stacked_params_mapping`).
- **`in_proj_a`/`in_proj_b`** are F32 (`install_gdn_ba_unquantized` already builds
  `in_proj_ba` unquantized) → `transform_plain`, applied *before* the stacked
  `in_proj_ba` `weight_loader(param, loaded_weight, shard_id)` at line ~249.
- **`in_proj_qkv`/`in_proj_z`** are quantized (`qweight`) → `transform_quant_rows`,
  applied to the packed `loaded_weight` before the stacked `in_proj_qkvz` load. **Verify
  first** how SGLang's GGUF loader hands these in (see §4.1).
- **`out_proj`** is quantized and needs a *column* permute → special-cased (§4.3).

Wire-in sketch inside the loop (name conventions confirmed §4.1: quantized tensors arrive
as `…qweight`; A_log/dt_bias are **bare** names; F32 projections/norms end in `.weight`):
```python
# after the conv1d unsqueeze block, before tie-embeddings:
if name.endswith(".qweight"):
    # in_proj_qkv / in_proj_z packed row reorder (out_proj is built unquantized, §4.3,
    # so it arrives as .weight and goes through transform_plain instead)
    loaded_weight = transform_quant_rows(name, loaded_weight)
elif not name.endswith(".qweight_type"):
    # F32 norms / A_log / dt_bias / in_proj_a / in_proj_b / conv1d / out_proj(unquant)
    loaded_weight = transform_plain(name, loaded_weight)
# NB: transform_* match on the HF module path; make them tolerant of a trailing
# `.qweight`/`.weight`/bare suffix (match on `…in_proj_z`, `…A_log`, etc.).
```
`transform_plain`'s `out_proj` branch (the `(H,2,16,128).permute(0,2,1,3)` op from §2) only
fires once §4.3 Option A makes `out_proj` unquantized. Never transform a `.qweight_type`.

---

## 4. The quantized-tensor complications (read before coding §3)

### 4.1 How does the GGUF loader present a quantized `loaded_weight`? — VERIFIED ✓
**Confirmed this session** (2026-07-23) from `sglang…/model_loader/weight_utils.py::
gguf_quant_weights_iterator` (lines 1190–1285) + direct `gguf.GGUFReader` probe of the
file. For a non-F32 tensor the iterator renames `…weight`→`…qweight` and yields
`param = torch.tensor(tensor.data)`; `tensor.data` is the **packed uint8** array shaped
`[out_features, packed_bytes_per_row]`. Measured on this GGUF:

| GGUF tensor | type | `data.shape` | dim 0 = |
|---|---|---|---|
| `blk.0.attn_qkv` (→in_proj_qkv) | Q6_K | **(8192, 3360)** | out (Q+K+V rows) ✓ |
| `blk.0.attn_gate` (→in_proj_z) | Q5_K | **(4096, 2816)** | out (V rows) ✓ |
| `blk.0.ssm_out` (→out_proj) | Q8_0 | **(4096, 4352)** | **out = hidden**; V is the *input* dim, packed into dim 1 ⚠ |
| `blk.0.ssm_alpha` (→in_proj_a) | F32 | (32, 4096) | v-heads ✓ |
| `blk.0.ssm_conv1d` | F32 | (8192, 4) | channels (Q+K+V) ✓ |

**Consequences (now certain):**
- **`in_proj_qkv` / `in_proj_z`: row indexing `qweight[row_perm]` is valid and cheap.**
  dim 0 is the output-feature axis; each row is a contiguous packed unit and llama.cpp
  quantizes along the *input* dim (now packed inside dim 1), so reordering whole rows never
  splits a quant block — same safety argument as `install_gdn_packed_gguf_loader_binding`.
  For qkv, permute only rows `4096:8192` (V); rows `0:4096` (Q,K) stay put.
- **`out_proj` is the awkward one, as predicted:** its V-head axis is the *input* dim,
  packed into dim 1 — **cannot** be row-indexed. Use §4.3 (Option A build-unquantized).
- **Incoming names carry `.qweight`** (not `.weight`) for quantized tensors, and A_log/
  dt_bias arrive as **bare** param names (no `.weight`). The transform dispatch must key on
  these (see §3 note). `qweight_type` scalars are yielded separately — do **not** transform
  them.

### 4.2 Interaction with the existing packed loader (patch #8)
`install_gdn_packed_gguf_loader_binding` splits `in_proj_qkv` into q/k/v shards via tuple
shard id `(0,1,2)` and fans the type scalar out. Your row permutation must be applied to
the **incoming source tensor** (the full `attn_qkv` / `attn_gate` qweight) **before** that
packed loader runs — i.e. in `load_weights` at the point you call
`weight_loader(param, loaded_weight, shard_id)`. Because the permutation only reorders
rows *within* the V block (and leaves Q/K rows in place), the subsequent q/k/v split at
`[0:2048], [2048:4096], [4096:8192]` still lands on the right boundaries.

### 4.3 `out_proj` — the one input-dim (column) permutation
`out_proj` `[H, V=4096]` must have its **input** columns reordered by `perm_head` at
128-col head granularity. Reordering columns of a packed row crosses the row's internal
block layout, so `qweight[:, col_perm]` is **not** valid on packed data. Two options:

- **Option A (recommended first cut): build `out_proj` unquantized.** Mirror
  `install_gdn_ba_unquantized`: add `install_gdn_out_proj_unquantized()` that overrides
  `Qwen3_5GatedDeltaNet.create_o_proj` (or the equivalent constructor) to pass
  `quant_config=None`, so `out_proj` loads as a plain `.weight`; then permute columns with
  the `transform_plain` out_proj tensor-op. **Cost:** F16 `out_proj` ≈ 32 MB/layer × 24
  linear layers ≈ **0.77 GB** extra VRAM. Acceptable on 12 GB for a correctness-first
  pass; optimize later if needed. *(Check the exact constructor name in
  `sglang.srt.models.qwen3_5.Qwen3_5GatedDeltaNet`; `create_ba_proj` is the analog.)*
- **Option B (memory-optimal, do later): packed block reorder.** `ssm_out` is **Q8_0**
  (block = 32 int8 + scale). A 128-col head chunk = exactly 4 whole Q8_0 blocks, so head
  chunks can be reordered without splitting blocks — but you must reorder the packed
  block-bytes + per-block scales together. More code; defer until Option A proves the math.

### 4.4 Universal fallback if packed row-permute proves unsafe
Force `in_proj_qkv` and `in_proj_z` unquantized too (same override pattern), permute as
plain tensors, and accept the VRAM cost (qkv F16 ≈ 64 MB/layer, z ≈ 32 MB/layer → ~2.3 GB
extra over 24 layers). Total worst-case unquantized-GDN-projection overhead ≈ ~3 GB — the
~6 GB model + this + KV cache still fits 12 GB at a modest context. Use this only if row
permutation on packed data can't be verified; it trades VRAM for certainty.

---

## 5. RMSNorm direction — VERIFIED ✓ subtract 1

**Confirmed this session** from the SGLang source used by the spike
(`sglang/srt/models/qwen3_5.py` + `sglang/srt/layers/layernorm.py`):

- `Qwen3_5` builds `input_layernorm`, `post_attention_layernorm`, `q_norm`, `k_norm`, and
  the final `model.norm` as **`GemmaRMSNorm`** (`qwen3_5.py:630-631, 835-841, 1198`).
- `GemmaRMSNorm` (`layernorm.py:648-691`) initializes `weight` to **zeros** (zero-centered)
  and its forward calls `gemma_rmsnorm(x, self.weight.data, eps)`, whose kernel applies
  `x_normed * (1 + weight)`. Its `_weight_loader` copies the loaded value straight into
  `weight.data` (and precomputes `gemma_weight = weight + 1`).
- Therefore the norm **expects the HF zero-centered `w`** and adds 1 internally. Feeding
  the raw GGUF value `1+w` would compute `1 + (1+w) = 2+w` → wrong. **→ subtract 1.** ✔

- `ssm_norm` (linear-attn gated RMSNorm) uses **`RMSNormGated`**
  (`qwen3_5.py:265`, from `fla/layernorm_gated.py:418`), which initializes `weight` to
  **ones** (standard RMSNorm, no offset). **→ do NOT subtract 1 from `ssm_norm`.** ✔
  (Matches the reference converter, which also excludes `ssm_norm`.)

Implementation note: these GemmaRMSNorm params carry the **custom `_weight_loader`**, which
`load_weights` picks up via `getattr(param, "weight_loader", …)`. Apply the `−1` to
`loaded_weight` **before** calling that loader (it then stores `w` and derives `w+1`).

---

## 6. Validation protocol (do this incrementally — one transform at a time)

The failure mode is silent, so validate against a **reference oracle** and enable
transforms **one at a time**, never all at once.

### 6.1 Build the oracle
Serve the *same* GGUF via the officially-supported GGUF runtime:
```bash
# llama.cpp is the blessed GGUF runtime for this arch (per the Ornith card)
llama-server -hf deepreinforce-ai/Ornith-1.0-9B-GGUF --port 8099 -c 4096
# (or point -m at the local Ornith-1.0-9B-UD-Q4_K_XL.gguf)
```
Prompt `"The capital of France is"` at `temperature=0`; capture the greedy continuation
and, if possible, top-k logprobs of the first token. **First re-confirm the GGUF file
itself is post-Feb-4** (the llama.cpp `key_gdiff` GDN fix) — if llama.cpp *also* produces
garbage, re-download the GGUF before trusting it as an oracle.

### 6.2 Unit-test the pure transforms (TDD, per REPO_RULES)
Before touching the server, test `ornith_gdn_transforms.py` in isolation:
- `perm_head` == `[0,16,1,17,…,15,31]` and is an involution-free bijection of `range(32)`.
- Round-trip: applying the converter's `reshape(2,16,…).permute` equals your index-based
  `_perm_rows` form on random tensors (assert exact equality) — this proves the packed
  row-index path matches the reference tensor-op path.
- `log(-x)` inverts `-exp(y)` on a random `y`.

### 6.3 Enable transforms one at a time, in this order
Toggle each via an env var (e.g. `ORNITH_FIX_NORM`, `ORNITH_FIX_ALOG`, `ORNITH_FIX_PERM`)
so you can isolate effects. After each, restart the server and compare to the oracle:

1. **A_log domain only** (`log(-x)`, no perm yet won't be correct alone but check it doesn't
   crash and shifts output).
2. **+ head permutation** (A_log, dt_bias, a, b, qkv-V, z, conv-V, out_proj) — this is the
   big one; expect a large jump toward coherence.
3. **+ RMSNorm −1** — expect the "first few tokens ok then degrades" symptom (§ converter
   README) to resolve into fully-stable output.

Success = greedy continuation matches (or closely tracks) the llama.cpp oracle and reads
as coherent English. If a step regresses, you've isolated the culprit transform.

### 6.4 Per-layer bisection (if still wrong after all three)
Hook `forward` to dump the hidden state after layer 0 (a linear-attn layer) and after
layer 3 (first full-attn layer); compare max-abs-diff to the same layers from the bf16
safetensors model (`unsloth/Ornith-1.0-9B` in native SGLang, run once on a bigger box or
CPU offload just to capture references). First divergence localizes the remaining bug —
likely mrope/partial-rotary on the full-attn path (§7) if the GDN path now matches.

---

## 7. Explicitly out of scope for this fix (but watch for)

- **mrope / partial-rotary on full-attention layers** (head_dim 256, `partial_rotary
  0.25`, `mrope_section [11,11,10]`, `rope_theta 1e7`). If GDN transforms land and output
  is *better but still subtly wrong*, this is the next suspect. The values are plumbed via
  `resolve_ornith_gguf_config.py` + `install_qwen3_5_text_config_hybrid_properties`;
  verify SGLang applies partial rotary to the correct 64 of 256 dims. Not part of the 3
  transforms.
- **Tokenizer reconstruction** (`gpt2` BPE, 248k vocab, `qwen35` pre). Low risk but cheap
  to exclude: compare token IDs for the prompt between SGLang and llama.cpp.
- **out_proj Option B** (packed block reorder) — a later VRAM optimization, not needed for
  correctness.

---

## 8. Deliverable / done criteria

- `deploy/sglang/native-ornith/ornith_gdn_transforms.py` (new, pure, unit-tested).
- `ornith_text_model.py::load_weights` calls the transforms (surgical edit per REPO_RULES).
- If Option A used: a new `install_gdn_out_proj_unquantized()` in `ornith_gguf_compat.py`,
  wired in `sitecustomize.py` alongside the others.
- Greedy output at `temperature=0` for `"The capital of France is"` is coherent and tracks
  the llama.cpp oracle.
- README "Remaining gap — output is not yet coherent" section updated with the resolution.
- DECISIONS.md records: the RMSNorm direction finding (§5), the quantized-`loaded_weight`
  layout finding (§4.1), and whether Option A or B was used for out_proj.

---

## 9. Key references

- Root-cause report: `deploy/sglang/native-ornith/RESEARCH-FINDINGS.md` §4b.
- Reference converter (the oracle for the transforms):
  `.scratch/vendor/gguf-to-nvfp4/scripts/step1_convert.py` → `gguf_tensor_to_torch()`
  (lines 102–160), `build_llm_mapping()` (38–67); its README §"Qwen3.5 GGUF → HF
  Conversion Pitfalls" (RMSNorm +1, A_log domain, value-head permutation).
- Current spike hook points: `ornith_text_model.py::load_weights` (172–262),
  `ornith_gguf_compat.py::install_gdn_ba_unquantized` (370–396, the unquantized-build
  pattern to copy for out_proj) and `install_gdn_packed_gguf_loader_binding` (299–367).
- GGUF ground truth: re-read with
  `cd deploy/sglang/native-ornith && NIXPKGS_ALLOW_UNFREE=1 devenv shell --impure -- uv run --locked --no-sync python <probe>`
  (`gguf.GGUFReader` on the `.gguf`).

> **NO SUBAGENTS. ALWAYS CONTINUE.** Work directly through PLAN.md; keep PROGRESS.md and
> CONTEXT.md current.
