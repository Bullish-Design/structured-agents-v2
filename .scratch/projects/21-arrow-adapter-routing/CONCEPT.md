# Project 21 — Arrow Adapter Routing

**Status:** concept / not started
**Depends on:** Project 20 (mixed-batch LoRA runtime, P2 fork), Project 17 (multi-LoRA router)
**Owner surface:** `src/structured_agents/llama_core/router.py`, `seq_routing.py`, the P2 fork graph

---

## 1. One-paragraph pitch

The multi-LoRA router today does not *choose* adapters — the caller declares them.
A `RouteRequest` carries `adapter="pydantic"`; `_run_seq_routed` looks up
`self._seq_index[r.adapter]` and calls `set_seq_adapter(ctx, seq_id, idx)` once per
wave (`router.py:376-378`). That is a **hard, static, per-sequence, label-driven**
assignment: the whole sequence gets exactly one adapter for its entire decode.

**Arrow** (from Ostapenko et al., *Towards Modular LLMs by Building and Reusing a
Library of LoRAs*) replaces that declaration with a **computed, per-token,
hidden-state-driven** decision, requiring **no training of a router** and **no
labels**. The routing weight for each adapter is derived directly from the adapter's
own weights (its top singular vector) and the token's hidden state. New adapters can
be hot-added to the pool and route automatically.

The punchline for our stack: **Arrow is the *soft* version of a hook we already
built.** The P2 fork's `seq_adapter_map` is a one-hot mask over the adapter pool.
Arrow replaces the one-hot `[0,1,0]` with a softmax `[0.2,0.8,0]` computed from `x`
at each layer. Same graph slot; the value flowing through it changes. This is
literally the "mask with parameter gravity" idea that motivated the investigation —
the gravity is the softmax gate.

---

## 2. Why this belongs here (and not vLLM/SGLang)

Standing repo rule ([[project17-custom-engine-not-vllm]]): project 17/20 is a custom
engine whose differentiator is the context-pool / seq-routed multi-LoRA router. Arrow
is the training-free routing brain that sits on top of the seq-routing capability we
already forked into llama.cpp. It is the natural next stop *after* mixed-batch
seq-routing, not a detour — because it extends the same masked/stacked `mul_mat_id`
path rather than introducing a new subsystem.

Related analogy that seeded this: CacheBlend does *selective* KV recompute only where
cross-block attention needs healing. Arrow does *selective* adapter application only
where a token actually points toward a specialist — the LoRA analog of selective
blending.

---

## 3. How Arrow runs during inference

### Step 0 — one-time prep: extract each adapter's "arrow"

A LoRA on a module is `ΔW = B·A` (scaled by alpha). Take the **top right singular
vector** of `ΔW` — one SVD per adapter, per target module, per layer. Call it `vᵢ`.
That vector is the adapter's "personality direction": the input direction it responds
to most strongly. Cache `{vᵢ}` alongside the adapter.

- Candidate set = the existing ordered pool (`_seq_pool`); pool index = routing id.
  No new adapter registration.
- Cheap: SVD of a rank-r `ΔW` is dominated by the `A` matrix (r × d_in); the top
  singular vector can be obtained from `A` (and `B`) without materializing the full
  `d_out × d_in` product.

### Step 1 — per token, score every arrow

A token's hidden state `x` enters a LoRA-adapted module. For each adapter compute:

```
gᵢ = |⟨x, vᵢ⟩|          # alignment of this token with adapter i's direction
```

Absolute value: a LoRA's *direction* matters, not its sign. Softmax the `gᵢ` over the
pool, keep top-k (typically top-1 or top-2). Result: a gate vector, e.g.
"0.8 pydantic, 0.2 general-python, 0 else."

### Step 2 — apply the adapters as a gated mix

```
y = W·x + Σᵢ gᵢ · scaleᵢ · Bᵢ·(Aᵢ·x)
```

Instead of one adapter's `B·A·x`, add a **token-weighted blend**. `class User(BaseModel):`
→ Pydantic arrow dominates. A plain `for` loop → general arrow wins. No caller hint.

---

## 4. Integration map — what exists vs. what Arrow adds

| Layer | Today (P2 static routing) | Arrow adds |
|---|---|---|
| **Python surface** (`router.py`) | `RouteRequest.adapter` required; `set_seq_adapter` once per wave | `adapter` optional; new `arrow` backend; when adapter is None the ctx runs "arrow mode" and picks internally — the router stops choosing |
| **Fork graph** (P2 `mul_mat_id` path) | one adapter's `B·A·x` per seq, hard index | dot-with-arrows → softmax → top-k → gated sum of adapters' `B·A·x`, computed *inside* the forward pass at every layer |
| **Prep / artifacts** | `llama_adapter_lora_init` loads B, A (`router.py:112`) | + SVD → cache `vᵢ` per module/layer; new artifact sibling to `fingerprint.py` |

### Critical constraint

Arrow **cannot** be driven from Python at the `llama_batch` level the way
`set_seq_adapter` is, because the gate depends on `x` *freshly at every layer*. It
must live in the graph. This is why Step 2 is a fork change, not a Python change.

### The primitive is already there

The P2b stacked `mul_mat_id` fusion ([[project17-p2b-fusion-trigger]]) computes
multiple adapters' contributions in one op. Arrow needs exactly that op — just fed
**data-dependent softmax weights instead of a 0/1 selection mask.** The stacked
fusion you'd trigger at high-K is the same kernel Arrow rides on. So Arrow is an
*extension* of the P2 masked path, not a new subsystem.

---

## 5. Build plan (delta from current state)

1. **Arrow extraction artifact** (library-side, doable now, no fork).
   - SVD each adapter's per-module `ΔW`; cache top singular vector `vᵢ`.
   - Sibling to `fingerprint.py`'s artifact registry; keyed by adapter + module + layer.
   - Validate against the tiny-LoRA eval set ([[project17-tiny-lora-eval]]): the
     ner-json / acrouter / brick-complexity adapters should produce distinguishable
     arrows (low pairwise cosine → good separability).

2. **Soft-gated stacked `mul_mat_id`** (fork-side, the real work).
   - Swap the one-hot `seq_adapter_map` lookup for a per-token
     `|⟨x,vᵢ⟩|` → softmax → top-k, reusing the stacked-adapter tensor already
     assembled in the P2b path.
   - Feed the arrows in as a constant tensor per module/layer.

3. **`arrow` backend in the router** (library-side).
   - Relax `RouteRequest.adapter` to optional.
   - Add `arrow` alongside `seq_routed` / `context_pool` in `_resolve_backend`;
     fail-closed like seq_routing (needs a fork lib that exports the arrow-gate
     capability; else fall back).

Steps 1 and 3 are library work available now. Step 2 is the fork change and is an
extension of the P2 masked path.

---

## 6. Open questions / risks

- **Where to gate:** per-module (query/key/value/MLP) arrows vs. a single
  per-layer arrow. Per-module is faithful to the paper but multiplies the SVD/dot
  cost; measure whether per-layer pooling is good enough.
- **Top-k cost:** top-1 is cheapest and often sufficient; top-2 blends but doubles
  the applied `B·A·x`. Sweep on the eval set.
- **Interaction with grammar-constrained decode:** Arrow gates on hidden state,
  independent of the logit-masking grammar path (`_sample_row`), so they should
  compose — but confirm the gate doesn't drift under heavy constraint.
- **Separability guarantee:** Arrow assumes adapters occupy distinct directions.
  Two adapters trained on overlapping tasks (python vs. pydantic) may have
  high-cosine arrows → weak routing signal. The extraction step should *report*
  pairwise arrow cosine as a health metric before anyone trusts the routing.
- **Scale semantics:** on `seq_routed` we already warn that per-adapter `scale`
  overrides are ignored (the fork applies built-in alpha). Arrow multiplies by a
  learned-free gate `gᵢ`; decide whether `AdapterSpec.scale` still means anything
  in arrow mode (likely: it pre-scales `Bᵢ·Aᵢ` before gating).

---

## 7. References

- Ostapenko et al., *Towards Modular LLMs by Building and Reusing a Library of LoRAs*
  (Arrow routing; training-free, SVD-derived per-adapter routing).
- Related composition methods surveyed for this line: Task Arithmetic, TIES-Merging,
  DARE, LoRAHub, Mixture of LoRA Experts (MoLE). Arrow is the per-token dynamic
  routing member of that family.
- CacheBlend (selective KV recompute) — the analogy that seeded the "selective
  specialist application" framing.
