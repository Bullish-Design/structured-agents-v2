# Project 19 — MoE + MoA Reactive Inference

*A single "thinking" loop model driving a harness of splice-able KV objects and a
multi-LoRA adapter pool, orchestrated by two small router models.*

Status: concept exploration. Date: 2026-07-30.

---

## 1. Overview / Thesis

Today's default answer to "make a small model punch above its weight" is either
(a) stuff more text into the prompt (RAG), or (b) train a bigger model. This
concept proposes a third axis: keep **one** small, strong central model in a
tight decode loop and give it a *learned harness* that can, between loop
iterations, (1) splice **precomputed key/value (KV) cache objects** into its
attention context and (2) dispatch side-tasks to a **multi-LoRA batched
inference pool** whose outputs are folded back into the response.

Two ideas from the literature are fused:

- **Mixture of Experts (MoE):** conditional computation — route each token to a
  small subset of specialized sub-networks instead of running the whole model.
- **Mixture of Adapters (MoA) / multi-LoRA:** many small LoRA adapters
  specializing a shared base, served together in one batch.

The novel claim is to lift both mechanisms *out of the weights and into the
inference harness*, and to add a third "expert" dimension the standard
architectures do not have: **experts over precomputed context (KV objects)**.
Instead of a gate choosing FFN experts per token inside a fixed graph, small
**router models** choose, per loop iteration, which cached KV objects to attend
to and which adapter-specialized side-jobs to launch. The central model is then
**trained to drive this harness** — to know when a splice or a side-job is worth
its latency — analogous to RL-for-tool-use (ReTool), but where the "tools" are
KV splices and adapter jobs rather than a Python interpreter.

Why this could matter: a MoE routes over *parameters* it already owns; this
routes over *context and specialization it can assemble on demand* from a
growing database, without retraining the base and without re-prefilling text it
has seen before.

### Honest framing note on "nanbeige-style"

The user's description says "nanbeige-style MoE base." Two corrections for
accuracy, because the grounding matters:

- **Nanbeige is not MoE.** The Nanbeige family (Nanbeige LLM Lab, BOSS Zhipin)
  — Nanbeige4-3B / 4.1-3B / 4.2-3B — is a *dense* transformer (~3.9B params,
  RoPE with ABF context extension to 64K). Its headline result is that careful
  data + training makes a 3B dense model beat much larger open models
  (Qwen3-8B/14B) on math, reasoning, and tool-use benchmarks (BFCL). So
  "nanbeige-style" is best read as *"a small, dense, unusually strong,
  tool-use-capable base"* — exactly the profile you want for the central loop.
- **What *is* genuinely relevant** is that **Nanbeige4.2-3B-Base uses a looped
  (recurrent-in-depth) transformer architecture** — the block stack is applied
  repeatedly. That "single looping model that does the smart thinking" framing
  in the concept maps directly onto Nanbeige4.2's looped design, and onto our
  own decode-loop harness. The MoE flavor in this proposal therefore comes from
  the *harness*, not the base weights. This is a defensible, deliberate design
  choice, not an accident.

We will keep the shorthand "the loop model" for the central model below.

---

## 2. Prior Art & Grounding

### 2.1 Mixture of Experts (conditional computation over parameters)

- **Switch Transformer** (Fedus et al., 2021) — top-1 routing: each token goes
  to exactly one FFN expert; simplest, cheapest gate.
- **Mixtral 8x7B** (Mistral, 2023) — top-2 of 8 experts per layer; the
  canonical open sparse MoE. Experts are large (~176M each).
- **DeepSeek-MoE** (2024) — two innovations we borrow conceptually:
  *fine-grained expert segmentation* (split each FFN into many small experts, so
  top-k picks from a much larger combinatorial space) and *shared-expert
  isolation* (a few always-on experts absorb common knowledge, freeing routed
  experts to specialize). DeepSeek's experts are ~8.7M each vs Mixtral's 176M.

The transferable lesson: **fine-grained + always-on-shared + sparse-routed** is
a better decomposition than a few big experts. Our KV objects and LoRA adapters
are the "fine-grained experts"; the loop model itself is the "shared expert."

### 2.2 Mixture of Adapters / multi-LoRA serving

- **S-LoRA** (Sheng et al., MLSys 2024) — serve *thousands* of concurrent LoRA
  adapters on one base via *unified paging* (one memory pool for adapter weights
  of varying rank + KV of varying length) and custom kernels over
  non-contiguous memory. Up to 4x throughput vs PEFT/vLLM.
- **Punica** — the SGMV (Segmented Gather Matrix-Vector) kernel that fuses many
  heterogeneous LoRA deltas into a single batched matmul, so a batch containing
  many *different* adapters runs in roughly the cost of one. S-LoRA extends
  Punica's kernels.
- **Mixture of LoRA Experts (MoLE / LD-MoLE / DynMoLE)** and **LoRAHub** —
  compose multiple LoRA experts. LoRAHub does *static* composition (fixed
  weights); MoLE-family methods add *dynamic, learned routing* over adapters at
  token/layer level. This is the closest existing analog to the "adapter pool
  router" here — but existing MoLE fuses adapter *outputs inside one forward
  pass*, whereas we dispatch adapters as *separate batched side-jobs* whose text
  output is spliced back.

### 2.3 KV cache reuse and modular / spliced KV (the crux)

- **PromptCache** (Gim et al., MLSys 2024) — precompute KV for frequently reused
  text *segments* and reuse them across prompts; a Prompt Markup Language assigns
  each module fixed position IDs so segments slot in at consistent positions.
  8x GPU / 60x CPU prefill speedups. This is the clearest precedent for a
  "database of reusable KV objects."
- **CacheBlend** (Yao et al., EuroSys 2025) — the key technique for *splicing KV
  computed in a different context*. Naively reusing a chunk's KV as a non-prefix
  is wrong because that chunk never cross-attended to the text now preceding it.
  CacheBlend reuses the cached KV but **selectively recomputes a small subset of
  tokens** (found by measuring KV deviation on the first few layers) to heal the
  cross-attention, recovering quality at a fraction of full prefill.
- **EPIC** (Efficient Position-Independent Caching) and **CacheClip / RedKnot /
  QCFuse** — a 2024–2026 line making cached chunks position-independent and
  cheaper to fuse (head-aware reuse, query-centric fusion). Confirms this is an
  active, unsolved-in-general area. EPIC's key math: RoPE composes, so a cached
  key realigns to a new position with a single rotation, R(δ)R(p₀) = R(p₁), and
  its **LegoLink** heals the per-chunk "attention sink" by recomputing only the
  first k tokens (k = 2–32, O(kN)).
- **KVLink** (arXiv 2502.16002, NeurIPS 2025) — the format we adopt (§6.7):
  store **unrotated** key states (W_k·x, no RoPE) and apply the global rotary
  at each token's correct full-sequence position at install time, making one
  blob valid at any position. Heals cross-chunk attention with trainable
  **link tokens** (KV computed at inference, custom attention mask) instead of
  selective recompute.
- **Research update (see §6.7):** the RoPE-shift kernel itself is *not*
  fork-blocked novelty — llama.cpp ships it natively (`llama_kv_cache_seq_shift`
  → `seq_add`, `build_graph_shift` in `src/llama-kv-cache.cpp`, used by
  context-shift, Self-Extend, and `n_cache_reuse`). What is genuinely missing
  upstream: position-selective *partial* splice, CacheBlend-style heal, and
  adapter-tagged admission (llama.cpp issue #26207 silently reuses KV across
  LoRA changes — see §6.7.4).
- **RAGCache** (Jin et al., 2024) — caches intermediate KV of retrieved docs in
  a knowledge tree across GPU/host memory, plus *speculative pipelining* to
  overlap retrieval with generation. Directly relevant to the KV-DB tiering and
  to overlapping router latency with decode.
- **StreamingLLM / attention sinks** (Xiao et al., ICLR 2024) — LLMs dump
  disproportionate attention onto the first few tokens ("sinks"); dropping them
  wrecks generation. And crucially: **RoPE positions must be assigned by
  position-in-cache, not position-in-original-text** — when you evict/splice, you
  re-index. This is the concrete mechanism our KV-splice must respect, and the
  reason our own repo has a `blend_by_reanchor` stub (see §7).

### 2.4 Routers, speculative decoding, retrieval-reactive decode

- **Speculative decoding** — a cheap drafter proposes tokens, the target model
  verifies in one pass. Establishes the pattern of a *small model feeding a big
  loop* with cheap verification, which the KV/task routers here echo.
- **REST** (Retrieval-Based Speculative Decoding) and **RAGCache's speculative
  pipelining / RASD** — draft tokens are *retrieved* from a datastore (suffix
  match / trie) rather than generated. This is decode-time retrieval reacting to
  the current generation — the same reflex the KV-router uses, but for drafts
  rather than KV objects.
- **kNN-LM** — interpolate the LM's next-token distribution with a distribution
  from nearest-neighbor datastore hits. The general "let a datastore bias the
  next step" primitive.
- **Learned harness / tool-use RL** — **ReTool** (ByteDance, 2025) trains a
  model via outcome-driven RL to decide *when and how* to invoke tools mid-
  reasoning; "Learning to Control LLM Agent Harnesses with Offline RL" (2026)
  learns harness-control decisions from rollout buffers with terminal rewards.
  These are the training templates for teaching the loop model to drive *our*
  harness (§5).

---

## 3. Architecture

Five components. The central model runs a normal autoregressive decode loop; the
harness intervenes at loop boundaries.

1. **Loop model (L)** — the dense, strong small base (Nanbeige-style; in our
   stack today, Ornith). Does all "smart thinking." Emits, alongside normal
   tokens, lightweight *harness-control signals* (see §3.6). It never blocks on a
   router: if nothing is spliced, it just decodes.
2. **KV cache database (KV-DB)** — a persistent, tiered store of *KV objects*:
   precomputed key/value tensors (+ metadata: token IDs, capture positions,
   base-frequency/RoPE params, engine fingerprint, an embedding for retrieval).
   Objects can be document chunks, tool-result summaries, code-tree nodes
   (our context-tree), or prior conversation spans.
3. **KV-router (R_kv)** — a *small* model / scorer that watches the live
   generation embedding + system/task state, retrieves candidate KV objects from
   the KV-DB (ANN over embeddings), scores them for usefulness under a token/
   latency budget, and produces a **splice plan** for the next loop iteration.
4. **Task-router (R_task)** — a small model that reacts to system/task/KV-DB/
   inference state and decides to launch **side-jobs**: dispatch a (prompt,
   adapter) pair to the multi-LoRA pool for "rapid context/whatever generation."
   Outputs are spliced back either as text tokens or as freshly minted KV
   objects.
5. **Multi-LoRA batched pool (A)** — an S-LoRA/Punica-style server holding many
   task adapters on the shared base, running heterogeneous adapters in one batch
   (our P2 mixed-batch fork). This is the "mixture of adapters" execution engine.

### 3.6 Harness-control signal channel

The loop model must be able to *ask* for harness actions cheaply. Options,
cheapest first: (a) a reserved set of control tokens / a small structured
"harness call" grammar the model can emit (grammar-constrained, like our router
MVP); (b) a learned auxiliary head predicting "splice-need" / "dispatch-need"
scalars per step; (c) fully implicit — routers act on hidden-state probes and the
model only *learns to leave room* for splices. We favor (a)+(b): explicit enough
to train with RL, cheap enough not to stall decode.

### Data-flow diagram

```
                          ┌───────────────────────────────────────────┐
                          │              SYSTEM / TASK STATE            │
                          │  (goal, tool results, budget, constraints)  │
                          └───────────────┬─────────────────────────────┘
                                          │ (state features)
        ┌─────────────────┐               │               ┌──────────────────┐
        │    KV-DB         │◄──ANN query───┤               │  MULTI-LORA POOL  │
        │  (tiered KV      │               │               │   (A: S-LoRA /    │
        │   objects +      │──candidates──►│               │   Punica batch)   │
        │   embeddings)    │               ▼               │  adapter_1..N     │
        └────────▲─────────┘        ┌─────────────┐        └────────▲─────────┘
                 │                   │  R_kv       │                 │
   new KV objects│                   │ (KV-router) │   dispatch      │ side-job
   (from side-   │                   └──────┬──────┘  (prompt,       │ result
    jobs, from   │                          │ splice        adapter) │ (text or
    decoded      │                          │ plan     ┌─────────────┴──┐  KV)
    spans)       │                          ▼          │  R_task        │
                 │        next-loop ┌─────────────────┐│ (task-router)  │
                 │        KV context│                 │└────────▲───────┘
                 └──────────────────┤   LOOP MODEL L  │         │
                                    │ (dense base,    ├─────────┘ state + gen
   tokens out ◄─────────────────────┤  decode loop,   │  embedding / control
                                    │  control head)  │  signals
                                    └─────────────────┘
                                    ▲ spliced KV (from R_kv)
                                    │ spliced text/tokens (from side-jobs)
```

Key property: **R_kv and R_task run concurrently with decode** (RAGCache-style
speculative pipelining). Their results land at the *next* loop boundary. If they
miss the boundary, the loop proceeds unspliced — the harness is best-effort and
never on the critical path for correctness.

---

## 4. The Reactive Loop Lifecycle (step by step)

Per decode iteration `t` (or per short window of tokens):

1. **Decode.** L runs one (or a micro-batch of) decode step(s) over its current
   KV context, emitting token(s) and, optionally, a control signal
   (splice-need / dispatch-need) plus a pooled hidden-state "generation
   embedding."
2. **Sense (async).** The generation embedding + current system/task state are
   handed to R_kv and R_task. This happens on a side stream so decode continues.
3. **KV retrieval + scoring (R_kv, async).** R_kv issues an ANN query into the
   KV-DB, gets candidate objects, and scores each for marginal usefulness under
   a remaining token/latency budget (a learned ranker; model2vec-style selector
   as a cheap first pass, as in our context-tree work — selector picks *which*,
   never produces KV). Produces a **splice plan**: {object_id, target position
   window, healing budget}.
4. **Task dispatch (R_task, async).** In parallel, R_task may decide a side-job
   is worth it: pick an adapter and a sub-prompt (e.g. "extract JSON schema for
   this API", "summarize retrieved doc", "draft a regex"), enqueue it into the
   multi-LoRA pool A. Jobs from many loop iterations (and other sequences) batch
   together — this is where MoA throughput lives.
5. **Splice at boundary.** Before iteration `t+1`, admitted KV objects are
   installed into L's context. This is the hard part (§6): objects were computed
   at *other* positions, so their K tensors must be **RoPE-re-anchored** to their
   new positions, and a small subset of tokens **healed** (CacheBlend-style
   partial recompute) so cross-attention is consistent. Preserve attention sinks.
6. **Fold-back.** Completed side-jobs return: their text is either (a) appended
   to L's stream as verified/pasted tokens (speculative-decoding-style: L can
   verify), or (b) converted into a *new KV object* and registered in the KV-DB
   for R_kv to splice — so an adapter's work becomes reusable context, not just
   one-shot text.
7. **Commit + learn signal.** Record what was spliced/dispatched, its cost
   (latency, tokens, healing FLOPs), and its effect on the trajectory. These
   become the reward-shaping records for §5.
8. **Loop.** Repeat until stop. Newly decoded spans can themselves be captured
   as KV objects (context grows the DB).

---

## 5. Training Strategy — teaching L to drive the harness

The base L is assumed already strong (Nanbeige/Ornith-class). The new skill is
*harness policy*: when to request a splice, when to dispatch a side-job, when to
just think. Staged plan:

**Stage 0 — Frozen-harness supervised warmup.** With hand-written / heuristic
routers, collect trajectories where splices/side-jobs *demonstrably* helped
(oracle: does having object X available improve next-span likelihood or task
reward?). Supervise L's control head to predict those requests. Cheap, stabilizes
the control channel before RL.

**Stage 1 — RL over harness actions (the core).** Follow ReTool / offline-RL-
harness-control: treat splice/dispatch/no-op as the agent's actions; the LLM
tokens are the environment interaction; reward = task success **minus** a real
cost term (latency, healing FLOPs, adapter-pool occupancy, KV-DB bandwidth). Use
outcome-driven RL (GRPO/PPO-style online, or advantage-weighted regression
offline from a rollout buffer as in the 2026 offline-harness-control paper).
Critically, **do not reward action patterns directly** — reward outcomes, so L
learns *efficiency*, not splice-spamming.

**Stage 2 — Co-train / distill the routers.** R_kv and R_task can be distilled
from L's own attention over spliced objects (which objects did L actually attend
to?) and from side-job utility. The routers become fast approximations of "what
L would have found useful," so they can run ahead of the boundary within budget.

**Stage 3 — Adapter-pool curation.** The set of LoRA experts is itself learned:
prune adapters R_task rarely dispatches; add adapters for recurring side-job
types (MoLE/LoRAHub-style composition as a fallback when no single adapter fits).

Reward-shaping specifics: a splice that L never attends to should be *penalized*
(wasted bandwidth), mirroring MoE load-balancing/auxiliary losses that prevent
router collapse. Attention-mass-on-spliced-object is a clean, cheap online
signal.

---

## 6. Hard Technical Problems & Open Questions

**(P1) KV splicing across positions — the central unsolved problem.**
A KV object computed at positions [100..140] in some other context cannot be
dropped into positions [520..560] here and be correct: (a) RoPE bakes absolute/
relative position into K at capture time, so keys must be **re-anchored** (rotate
cached K by the position delta) — feasible with RoPE but needs raw K-buffer
access; (b) the object never cross-attended to the *current* preceding context,
so its V/attention is stale — CacheBlend shows you must **recompute a subset of
tokens** to heal it, which costs FLOPs and needs a per-object healing-budget
policy; (c) **attention sinks** — StreamingLLM shows the first tokens hoard
attention; a splice must not displace/duplicate sinks or generation degrades.
Our repo already hit exactly this wall: `blend_by_reanchor` is a documented
NotImplementedError because the stable llama.cpp C API does not expose the KV
buffer for RoPE-shift; only `blend_by_redecode` (partial re-prefill) works today.
**Open:** is there a healing budget that makes re-anchor + heal reliably cheaper
than re-decode on our GPUs? (Our own break-even data, §7, says "prefill is cheap"
— so the bar is high.)

**(P2) Router latency vs. decode speed.** A splice/dispatch decision is only free
if it finishes before the next boundary. ANN query + rank + RoPE-shift + heal
must overlap decode (RAGCache pipelining). If the routers are too slow, the
harness stalls the loop it was meant to accelerate. **Open:** how small can R_kv
be and still rank usefully? Is a model2vec selector + tiny cross-encoder enough?

**(P3) Consistency & correctness under splicing.** Every splice mutates L's
context mid-generation. We need: idempotent, reversible splices; guarantees that
a bad splice degrades gracefully (best-effort, never corrupts); no cross-sequence
leakage in the batched pool (S-LoRA isolation). Our project-18 lesson — *"a
successful byte-copy is not a successful restore; only a matching deterministic
continuation counts"* — applies directly: every splice needs a continuation-
equivalence test, not just "it loaded."

**(P4) Is it ever a net win?** Project 18's break-even benchmark is sobering:
moving whole-context state blobs (69–188 MB) was **5–9x slower** than just
recomputing prefill on a 3060, and native in-memory KV only crossed over around
~256 tokens. Splicing *small* KV objects with *partial* healing is a different
regime, but the burden of proof is real: splice+heal must beat re-prefill of the
same tokens. This must be measured, not assumed.

**(P5) Router collapse / load balancing.** Like MoE gates, R_kv/R_task can
collapse onto a few objects/adapters. Needs auxiliary balancing losses and the
attention-mass penalty from §5.

**(P6) Credit assignment.** With two async routers and a fold-back channel, which
component gets credit/blame for a good/bad trajectory? Offline advantage-weighted
methods help, but attribution across L / R_kv / R_task is genuinely hard.

**(P7) KV-DB freshness & positional metadata.** Objects captured under one RoPE
base frequency / context config are not portable to another (our fingerprinting
work). The DB must key on engine fingerprint + capture positions, and invalidate
aggressively.

---

## 6.5 Expert Bundles: Distilled Adapters over the Depth Loop

The architecture above treats the loop model L and the adapter pool A as separate
components. This section develops a design refinement that fuses them: instead of
one static base looped `n` times, L wears **a different LoRA adapter on each pass
through the layer stack** — a "LoRA hat" per ponder-step — and each hat carries
its own knowledge. This turns the depth loop itself into a mixture-of-adapters
axis and gives us a clean unit of composition, the **expert bundle**.

### 6.5.1 The loop is over depth, not sequence position

The critical framing: Nanbeige4.2's looped-transformer design (and the broader
recurrent-in-depth line — **Universal Transformer** (Dehghani et al., 2018) with
its per-position **Adaptive Computation Time** halting (Graves, 2016), and
**CoTFormer** (Mohtashami et al., 2023) which drives extra layer applications as a
latent chain-of-thought with budget-adaptive compute) re-applies the *same block
stack* to think harder about the *same tokens*. The recurrence is over **depth /
ponder-steps**, not over sequence position (it is not an RNN scanning left to
right). Each extra pass is another increment of "thinking," not another token.

The refinement: make each pass wear a **different** adapter. Pass 1 might be a
"frame the problem" hat, pass 2 a "critique / find the flaw" hat, pass 3 a
"commit / finalize" hat. Each hat is a *cognitive mode* over the shared latent,
and the sequence of hats is a fixed (or routed) program of thought executed
in-depth before the next token is emitted. This is a **mixture-of-adapters over
the recurrence/depth axis**, and it is a genuinely different routing key from the
prior art:

- **MoLE / LoRAHub / S-LoRA** (§2.2) route adapters **per-request** or
  **per-token** — one adapter (or a fused blend) chosen for the input.
- **Here routing is per ponder-step**, keyed on the **loop index** (a fixed
  program) or on the **hidden state** at the start of that pass (a learned gate,
  like ACT's halting unit but choosing *which hat* rather than *whether to stop*).

The same P2 mixed-batch fork that batches adapters *by request* becomes the
substrate for swapping adapters *by loop-step* — the mechanism is identical, only
the index changes (see §6.5.4).

### 6.5.2 Expert bundle = skill + knowledge

Define an **expert bundle** as a pairing of a **skill** (the adapter weights —
the cognitive mode) with its **knowledge** (the facts/context that mode needs).
The skill is always the LoRA; the knowledge can be attached two ways.

**(1) Fixed-position read-only KV reference prefix (PromptCache-style).** Attach
the knowledge as a precomputed KV block the hat attends to, exactly as PromptCache
(§2.3) reuses KV for reusable text segments at fixed position IDs. Two consistency
rules are non-negotiable:

- **(a) The KV must be precomputed under that hat's own adapter weights.** Because
  `K = x·W_k` and `V = x·W_v`, and the LoRA modifies `W_k`/`W_v` (or the layers
  that feed them), the KV for hat *i* is *not* interchangeable with hat *j*'s or
  with the base's. A bundle's reference KV is captured while wearing that bundle's
  hat, or the keys/values it exposes are simply wrong for the attention that will
  read them.
- **(b) It must stay a fixed-position, read-only prefix — not spliced into the
  mutable running sequence per pass.** A fixed read-only prefix at stable position
  IDs is the safe, PromptCache-legal case (append along one chain — our
  *exact single-chain reconstruction* result, §7). The moment you splice the
  knowledge into the *middle* of the mutable running sequence on each pass, you
  reintroduce the **CacheBlend cross-attention-consistency problem** (§2.3, P1):
  the block never cross-attended to the freshly-decoded context now around it, and
  healing it needs the RoPE-shift + partial-recompute path that is **fork-blocked
  on the stable llama.cpp API** (`blend_by_reanchor` is a NotImplementedError,
  §6/§7). So: reference prefixes yes, per-pass mid-sequence splices no.

**(2) Distill the context into the adapter weights (preferred where feasible).**
Rather than carry a KV prefix at all, **bake the knowledge into the LoRA** via
**context distillation** — **Snell et al., "Learning by Distilling Context"
(2022)** and the compression variant, **Mu et al., "Learning to Compress Prompts
with Gist Tokens" (2023)**. The recipe:

- **Teacher:** base + the context in the prompt (the hat's knowledge spelled out).
- **Student:** base + this hat's LoRA, with the context **removed** from the input.
- **Objective:** match output distributions (token-level **KL on logits**) over a
  diverse query set that exercises the knowledge.

The student learns to behave *as if* it had read the context, with the context
gone at inference. The payoff is decisive for our cost model: **zero KV at
inference** — the knowledge becomes **parametric**, folded into the same low-rank
matmul that already carries the skill. This dissolves *both* consistency traps of
option (1) at once: rule (a) is moot (no KV to keep adapter-consistent) and rule
(b) is moot (no block to splice into a mutable sequence), and it erases the KV
storage and attention-length cost entirely.

### 6.5.3 Honest tradeoffs of distillation, and the hybrid

Distillation is not free knowledge:

- **Low-rank capacity ceiling.** A rank-`r` LoRA has a finite budget. Style,
  procedure, and *small* fact sets internalize well; a *large factual corpus* does
  not fit — you cannot distill a 200-page manual into rank-16.
- **Verbatim recall degrades.** Attention over a KV prefix can *quote exact
  tokens*; distilled weights capture the **gist**, not the surface form. Anything
  needing literal reproduction (API signatures, legal text, exact identifiers)
  wants KV, not weights. (This is Gisting's own boundary: compression trades
  fidelity for footprint.)
- **Static.** A distilled hat is frozen — a context change means a **retrain**,
  whereas a KV prefix is a cheap recompute. Volatile knowledge should not be
  distilled.
- **Training cost moves up-front.** You pay once, offline, per hat — acceptable
  for stable modes, wasteful for one-shot context.

**Conclusion — hybrid.** Distill the **stable style/procedure/small-facts** into
the weights; keep **large, volatile, or verbatim-recall** knowledge as a
fixed-position read-only KV prefix (option 1). In practice **many hats end up
pure-adapter, zero-KV** — the "frame," "critique," and "commit" cognitive modes
are procedure, not corpus — and those are the **cheapest inference path** we have:
no KV to store, re-anchor, heal, or attend over. The KV machinery is reserved for
the minority of bundles that genuinely need a corpus.

### 6.5.4 Cost model: unmerged adapters, indexed by loop-step

The per-pass swap must be cheap or the loop economics collapse:

- **Keep adapters unmerged.** Apply `x·W + (x·A_i)·B_i` as an *extra low-rank
  matmul* per pass — the S-LoRA/Punica path (§2.2) — never fold `A_iB_i` into `W`.
  **Merging per pass would dwarf the loop**: a merge is a full-weight write, orders
  of magnitude more work than the low-rank delta, and it would have to be undone
  before the next hat. Unmerged, the hat swap is just selecting a different
  `(A_i, B_i)` for the same batched delta kernel.
- **The P2 mixed-batch LoRA fork is the right substrate — re-indexed.** P2 already
  runs heterogeneous adapters in one batch keyed **by request**
  (`llama_set_seq_adapters`, the P2b fusion trigger `2Kr/d`, §7). The only change
  for hats is the **index: by loop-step instead of by request.** Same SGMV-style
  batched delta, same fork, different selection vector. This is why the refinement
  is cheap to reach from where the repo already is.

Two caveats that constrain training:

- **Looped weights must be stable under re-application.** A block stack applied
  `n` times must not diverge/explode across passes — the same stability property
  Universal Transformer/CoTFormer rely on. Hats perturb the pass, so each hat must
  preserve that stability, not break it.
- **Hats must be trained *unrolled*, with an end-of-loop loss.** A hat trained in
  isolation will not compose in sequence. Train the full program
  (frame→critique→commit) unrolled through the depth loop with the loss applied at
  the **end of the loop**, so the hats learn to hand off to each other — pass 2's
  input is pass 1's output *under pass 1's hat*, and the gradient must flow through
  the whole ponder chain. This is the analog of BPTT for a depth loop, and it is
  what makes a *sequence* of cognitive modes coherent rather than three unrelated
  adapters.

### 6.5.5 Relationship to the rest of Project 19

Expert bundles slot directly onto existing scoping. The **distilled, zero-KV
hats** are the ideal case the P4 break-even worry (§6) has been pushing toward all
along — no KV to move means the "splice must beat re-prefill" burden simply does
not apply to those bundles. The **KV-carrying hats** reuse the *safe* splice
(fixed-position read-only prefix = single-chain append, already token-exact and
GPU-verified, §7) and pointedly **avoid** the unsolved cross-chain
`blend_by_reanchor` path. The pool A becomes a library of bundles indexed by
ponder-step; R_task (§3) can still dispatch bundles as side-jobs, but the depth
loop now consumes them internally, per pass, via P2. In short: expert bundles are
how the "mixture of adapters" of this proposal becomes a *mixture of cognitive
modes over depth*, with knowledge preferentially compiled into weights so the
cheapest hats need no harness at all.

---

## 6.6 Per-hat partners: prefetchers, prefix-stagers, and function dispatch

The architecture so far has **one global `R_task`** (§3.4). This section develops a
refinement: give **each depth-loop hat (§6.5) its own partner adapter** in a second
pool that shares the *same common small-model core* (the P2 mixed-batch substrate —
partners are just more unmerged `(A_i, B_i)` deltas in pool A). Each partner is the
mode-specific dispatcher for its hat: "what does *this cognitive mode* need pulled,
warmed, or called." Splitting the global router into per-hat partners sharpens credit
assignment (P6 — a request is attributable to a mode, not to a global guess) and
narrows each partner's input distribution (P5 — less collapse). It costs no new engine
primitive; the partners batch alongside the hats in the existing fork.

The design contract that keeps this buildable **without waiting on the unsolved
`blend_by_reanchor` kernel (P1/Spike B)**: partners may do three things, and only
three. The first two are free or safe today; the third (function dispatch) folds back
by a *different, cheaper* mechanism than KV splicing.

### 6.6.1 The timing invariant (why "end of pass" is the wrong frame)

A ponder-pass through the block stack is milliseconds; a partner action (ANN query,
adapter decode, external tool round-trip) is orders of magnitude slower and lands
**tokens later**, not at the depth-pass boundary. So partners are **async producers**,
never synchronous end-of-pass returns. A partner request issued during hat *i*'s pass
at token *t* has its effect staged for hat *i*'s pass at some **future token step**,
routed in by R_kv/the fold-back path — never blocking the loop it was meant to help
(the §4 best-effort property). "Warm it for next turn," not "return it this turn."

### 6.6.2 What a partner may do

| Mechanism | Feasibility | Why |
|---|---|---|
| **(1) Prefetch / pin** KV to VRAM (cache warming) | **Free — do now** | No splice, no P1; Stage-0 supervisable |
| **(2) Stage a read-only prefix** for the hat's next turn | **Safe — available today** | = single-chain append, token-exact, GPU-verified (§7) |
| **(3) Dispatch a function call**, fold result back | **Buildable now** | Fold-back by token-verify / prefix, not mid-sequence splice |
| *(mutate the running middle)* | **Gated on Spike B** | = `blend_by_reanchor`, still fork-blocked — **out of scope for partners** |

**(1) Prefetch / pin.** The partner predicts "hat *i* is about to want object X" and
issues the ANN lookup + tier promotion (host→VRAM, or recompute-and-hold) *ahead* of
the pass that needs it — RAGCache-style speculative pipelining (§2.3). Nothing is
installed; R_kv still decides admission later. Zero correctness surface: a
mispredicted prefetch wastes bandwidth, not correctness (the §5 attention-mass penalty
already prices wasted pulls). **Trainable in Stage 0, not Stage 1** — the oracle is a
cheap binary label: *did the object this partner warmed get admitted by R_kv within K
tokens?* No RL needed. This alone justifies the per-hat partner.

**(2) Stage a read-only prefix.** The partner grows a **fixed-position, read-only KV
prefix** the hat attends to next turn — §6.5.2 option (1), the *safe* splice. Because
it targets the **prefix at stable position IDs**, not the mutable running middle, it is
the single-chain append this repo has already proven **token-exact and GPU-verified**
(§7) — it sidesteps `blend_by_reanchor` entirely. Two non-negotiable rules carry over
from §6.5.2:
- **(a) Adapter-tagged.** The prefix must be KV-computed under *that hat's own adapter
  weights* (`K = x·W_k`, and the hat perturbs `W_k`/`W_v`). Objects are tagged by hat;
  R_kv may only stage object into hat *i* if captured under hat *i* (or under base, if
  the small inconsistency is accepted). This multiplies KV-DB storage per shared corpus
  and is a fingerprint-invalidation surface (P7) — bookkeeping, not a wall.
- **(b) Prefix-only, never the middle.** Partners **grow the read-only prefix; they do
  not mutate the running sequence.** The moment an insert lands in the grown middle it
  is the CacheBlend cross-attention problem (P1) — off-limits for partners by contract.

**(3) Function dispatch — kept, and cheaper than context splicing.** The
"kick off functions / whatever might be useful" half is retained and is *more*
tractable than KV context, because the fold-back mechanism is different. The partner
emits a **structured function call** (grammar-constrained, like the router MVP / §3.6
control channel); it executes async on a side stream; the **result** returns and folds
back by one of two splice-free paths:
- **(3a) Verified token append (speculative-decoding-style, §2.4).** The result text is
  proposed into L's stream and L *verifies* it in its normal forward pass — the same
  cheap-drafter/verify pattern as REST/spec-decode. No KV re-anchor: the tokens enter
  at the live head position, which is always position-consistent by construction.
- **(3b) Freshly-minted read-only prefix (→ path 2).** The result is captured as a new
  adapter-tagged KV object and registered in the KV-DB, to be staged as a prefix on a
  later turn. Reuses path (2)'s safe append; never touches the middle.

Either way, **function fold-back never invokes `blend_by_reanchor`.** This is why the
function-calling ambition survives Spike B being unsolved: a returned *tool result*
enters as verified tokens or a prefix, both of which are position-consistent by
construction. The dispatch decision is per-hat (a "verify" hat dispatching a unit-test
run; a "retrieve" hat dispatching a search), keyed on the hat's start-of-pass hidden
state.

### 6.6.3 Training the partners

- **Prefetch head (path 1):** Stage 0 supervised, offline oracle (admitted-within-K).
- **Prefix-stage head (path 2):** Stage 0 supervised on the same admission oracle, plus
  the §5 attention-mass signal (did the hat actually attend to the staged prefix?).
- **Function-dispatch head (path 3):** this is the one that genuinely needs **Stage 1
  RL** — the payoff of a tool call is displaced in time and gated on external latency,
  so it can't be backprop'd through the unrolled depth loop (P6). Reward = task success
  minus call cost (latency, tool occupancy), outcome-driven (ReTool template, §2.4/§5).
  Do **not** reward dispatch patterns directly, or partners learn to call-spam.

The partners are trained **jointly, unrolled** with their hats (§6.5.4 BPTT-for-depth)
for the *decision to request*; the *async effect* of paths 1–2 is supervised offline
and of path 3 by RL. Split the objective by mechanism — do not try to backprop a
wall-clock race.

### 6.6.4 Net

Under one discipline — **partners only (1) warm cache, (2) grow an adapter-tagged
read-only prefix, and (3) dispatch functions that fold back as verified tokens or a new
prefix — never mutate the running middle** — the entire per-hat-partner concept,
*including function calling*, is buildable on the existing P2 pool + context-tree splice
**without** the unsolved kernel, and is trainable with cheap offline labels for paths
1–2 and outcome RL for path 3. The only piece still parked behind Spike B is the
"large cacheblend insert into the live context" ambition — and by this contract the
partners never need it.

---

## 6.7 KV-DB object format: unrotated keys, quantized tiers, KVLink healing

Research B (`.scratch/projects/19-moe-moa-reactive-inference/RESEARCH-B-PRIOR-ART.md`, prior-art deep-research with all claims adversarially verified) settled the KV-object design. Decision: **KVLink-style unrotated-K blobs, quantized at rest, f16 in the live cache, rotation only ever in float.**

### 6.7.1 Why unrotated keys (KVLink-style), not post-hoc re-anchor

- **Store `W_k·x` / `W_v·x` without RoPE baked in.** Position is applied at install: dequantize → apply R(p_install) in float → write f16 into the live cache. No stored value is ever rotated, so quantization error and rotation error never interact (RoPE rotation is orthogonal/norm-preserving, so it would not amplify stored error anyway — with unrotated storage this is structural, not incidental).
- **Position-agnostic blobs.** One blob is valid at any install position (EPIC's R(δ)R(p₀) = R(p₁)); R_kv can stage an object and admit it wherever it lands (§6.6.1). This is the property the whole "KV-DB as an asset" story needs.
- **V never rotates** — position-free by construction; installed by dequant + copy.

### 6.7.2 Blob layout and storage tiers

Per KV object, in order: **(1) f16 header** — first-k sink tokens per chunk (k = 16–32), unquantized; attention dumps onto these (StreamingLLM / EPIC sink effect), so they are the highest-stakes tokens in the chunk. **(2) Adapter-tag** — the hat identity under whose weights the blob was captured (§6.6 rule (a)); R_kv admits into hat *i* only if tagged *i* (or base). **(3) Quant-format field** — q8_0/q4_0 + version; part of the P7 fingerprint, invalidate on format change. **(4) Payload** — quantized unrotated K + V blocks for the remaining N−k tokens.

| Tier | Format | Purpose |
|---|---|---|
| Hot (VRAM pin) | f16, rotated at install | §6.6 path-1 prefetch target; ready to attend |
| Cold (disk / DB) | q8_0 default; q4_0 for large objects | 2–4× smaller; dequant on promote |
| Sink header | f16 always | first-k tokens per chunk |

Header overhead vs all-quantized (f16 = 2 B, q8_0 ≈ 1.06 B, q4_0 ≈ 0.56 B per element):

| chunk N | header k | vs all-q8_0 | vs all-q4_0 |
|---:|---:|---:|---:|
| 1024 | 16 | +1.4% | +4.0% |
| 1024 | 32 | +2.8% | +8.0% |
| 256 | 16 | +5.5% | +16% |
| 256 | 32 | +11% | +32% |

At our use case (large objects, hundreds+ tokens) the header is noise. If V stays f16 in both designs, halve every number. If small chunks + q4_0 make it expensive, shrink the header to k = 0 for healing and keep a minimal header for sink integrity — the link-token heal (§6.7.3) needs no precision header at all.

Load-bearing rules:
- **Quantized at rest; f16 in the live cache; rotation only ever in float.** llama.cpp issue #5652 crashes because it rotates a *still-quantized* q4_0 K tensor in place (`GGML_ASSERT: ggml.c:12646`). Our pipeline dequantizes before rotating, so it never touches quantized memory with the RoPE kernel. The runtime cache stays f16 so llama.cpp's own context-shift (if triggered at window fill) also never touches quantized memory.
- **q8_0 ≈ lossless in practice** (2× saving); q4_0 has measurable degradation — reserve for cold/large objects. The heal check-layer comparison (if CacheBlend-style) must run in float, or quantization noise pollutes the top-k deviation selection.
- **Adapter-tag and quant-format travel with the blob** — both are admission checks and both are P7 fingerprint fields.

### 6.7.3 Heal mechanism: link tokens over CacheBlend recompute

Two heal families exist; the choice is driven by whether we are willing to train.

- **CacheBlend recompute-heal (training-free):** fully recompute layer 1, partially layer 2, compare V against precomputed → recompute the top-k High-KV-Deviation tokens (~10–15%). OSS in LMCache (vLLM-side). Works untrained, but pays recompute per install and its deviation signal needs float precision. Positions are handled by *recompute*, not shift — complementary to the re-anchor story, not a substitute for it.
- **KVLink link tokens (trainable) — preferred.** K = 5 trainable tokens per chunk, KV computed fresh at inference, custom attention mask (links attend to all prior; document tokens keep causal attention). Fresh in-context notes bridge cross-chunk attention with no per-install recompute and no precision header. Requires model fine-tuning — already on the table here (§6.5.4 / §6.6.3 unrolled joint training), so the cost is paid once and shared.
- **EPIC LegoLink (fallback, training-free):** recompute only the first k tokens per chunk (O(kN)) to kill the sink effect at chunk starts. Pairs naturally with the f16 sink header — the header *is* the unquantized sink set.

### 6.7.4 What the research changed

- **The "impossible" kernel is upstream.** `llama_kv_cache_seq_add` / `build_graph_shift` already re-applies RoPE to cached K without re-processing tokens; `n_cache_reuse` already RoPE-shifts a non-prefix matching slice. The fork work is *exposing* partial-position splice + heal + adapter-gating, not inventing rotation. SPIKE-B-RESULTS.md already noted `llama_memory_seq_add` + `llama_memory_can_shift` in `llama.h` (~749/782) can re-anchor a whole restored sequence today.
- **Adapter-tagged admission is novel *and required*.** llama.cpp issue #26207 (2026-07-28): llama-server reuses prompt-cache KV across different LoRA adapters on prefix match → silently contaminated output. Upstream does not key cache on adapter config; §6.6 rule (a) is therefore not conservative, it is the correct behavior upstream is missing.
- **Verify before betting:** (a) Nanbeige's rope type — llama.cpp hard-disables KV shifting for M-RoPE / interleaved / reasoning-token architectures (discussion #24944); looped/recurrent bases need the check. (b) Runtime `--cache-type-k` stays f16 (issue #5652). (c) `--cache-reuse` falls back to full reprocessing when the shared content is not a contiguous block (discussion #22354) — no heal path in-tree, confirming the heal must come from the CacheBlend/KVLink playbook.
- **References:** MiniPIC (IBM/vllm fork) is the minimal reference for deferring RoPE to attention time (<100 LOC); KVLink is the training-involved variant that fits this project's regime; CacheClip (aux-model-guided heal selection) is the alternative if the link-token training proves fragile.

### 6.7.5 KV-cache-specific quantization (beyond q8_0/q4_0)

Deep dive: `RESEARCH-C-KV-QUANTIZATION.md`. The DB blob at rest is **engine-agnostic** (only a dequant kernel is needed; llama.cpp cache types are plain `ggml_type` block formats — no per-channel/per-token). Research formats are therefore available as storage formats even without engine support. Ladder: **(a) now** — q8_0 default / q4_0 cold; **(b) next, best fit** — KVQuant-style **per-channel K** (we can, because our keys are unrotated — RoPE mixing is what normally breaks per-channel K), **per-token V** (V outliers are per-token), **non-uniform NF4 LUT** → ~3–4 bits at q8_0 quality; **(c) aggressive** — KIVI-style mixed window (extend the f16 header to first-k *and* last-k tokens) or GEAR-style low-rank+sparse residual → 2–3 bits. The mixed-window principle independently validates the f16 sink header (§6.7.2). Rules unchanged: rotation only in float, heal check-layers in float, P7 fingerprint includes quant format + hyperparameters, acceptance = continuation-equivalence (output error, not weight distance — MHA2MLA's finding, RESEARCH-D).

### 6.7.6 Future payload: MLA latents ("cards")

Deep dive: `RESEARCH-D-MLA.md`. MLA caches a per-token **latent** (position-free by construction) instead of K/V; for an MLA model the DB card and the live cache cell are the **same unit** — install = dequant → memcpy, no rotation, no per-channel K complexity, ~14× smaller objects. Not a format: it requires an MLA-native or **converted** base. Conversion is now practical: TransMLA (GQA→MLA, ~6B tokens), X-EcoMLA (AMD, 1B/3B models released), MHA2MLA (MHA→MLA, 0.3–0.6% of data). **To verify: Nanbeige's attention type** — GQA → TransMLA recipe, MHA → MHA2MLA recipe. Caveats: adapter-tagging gets *more* critical (hats perturb `W_UK`/`W_UV`; the dictionaries join the P7 fingerprint), and installing foreign cards still needs a fork install primitive (but with trivial math). Related but orthogonal: Nanbeige 4.5 previews *native* depth-attention (mHC) — `NANBEIGE-45-PREVIEW.md` — the model-native answer to cross-depth KV validity.

---

## 7. Relationship to this repo's existing work

This proposal is not greenfield — it is the *union* of four things Project 17/18
already built or scoped, plus a training loop on top. Mapping:

- **Loop model L → Ornith on the llama.cpp inference lab (Project 17).** We
  already run Ornith (a hybrid GatedDeltaNet + attention arch) on a custom
  llama.cpp build via CFFI, GPU-verified on 3060s. The "custom engine, not vLLM"
  memory is load-bearing: *multi-LoRA here is our context-pool router, and we do
  not propose vLLM/SGLang for proj17.*

- **KV-DB + KV-router → the context-tree KV deltas + prefix cache (Project 17).**
  `node_delta*.py` / `node_blend_live.py` already implement per-node KV objects
  over an LSP codebase tree: capture each node's KV once, reconstruct any
  ancestor chain with **zero prefill**. That *is* a KV object database with a
  tree index. `model2vec` is already used as the *selector* (rank which pull-ins
  to admit under budget) — that is precisely R_kv's cheap first pass.
  - **Exact single-chain reconstruction works** (token-exact vs cold prefill,
    GPU-verified). That is the "safe" splice: append along one ancestor chain.
  - **Cross-chain splice is the open P1 above.** `blend_by_redecode` (partial
    re-prefill) is implemented and correct; `blend_by_reanchor` (CacheBlend-style
    RoPE-shift + heal, true zero-prefill splice) is a **stub** blocked on
    KV-buffer access the stable C API doesn't expose (needs a native ggml shim or
    upstream `kv_rope_shift`). **This is the single most important dependency for
    Project 19.**

- **Persistent KV storage → Project 18.** `PersistentPrefixCache` (atomic fsynced
  blobs + SQLite checkpoint-boundary index, exact-continuation tested across
  process restart) is the durable tier of the KV-DB. Its lessons constrain us:
  *correct ≠ faster*; whole-state blobs are mostly dead weight (~0.99 MB/token
  was unused `scores`); a native codec (`llama_state_get_data`, no score buffer)
  is the smallest justified next step. Project 19's KV objects must be *small and
  partial*, not whole-context blobs, or P4 kills it.

- **Multi-LoRA pool A → the P2 mixed-batch LoRA fork (Project 17).** The
  `patches/p2-mixed-batch-lora.patch` + `llama_set_seq_adapters` per-sequence
  adapter path is exactly S-LoRA/Punica-style heterogeneous batching in our
  engine. The P2b fusion-trigger memory (switch masked→stacked `mul_mat_id` only
  when K~dozens/high rank; formula 2Kr/d) is the throughput knob for the pool.
  Ornith LoRA runtime is a confirmed GO (P0 gate cleared: converted GGUF LoRA
  applies to the hybrid arch at runtime).

- **The four reactive-inference gaps memo is the Project 19 backlog.** It already
  enumerates what's missing for "internally reactive inference": (1) **no
  embedding/ANN infra** — the biggest build, and R_kv's prerequisite; (2)
  **batched decode loops bypass the `on_logits` middleware** — must be wired in
  for any reactive biasing on branches; (3) **per-seq LoRA needs the P2 fork**
  (have it); (4) **in-place seq fork not wrapped** (`llama_memory_seq_cp`
  unwrapped; true re-anchor is the P1 stub). Project 19 is essentially "close
  these four gaps and put a trained policy on top."

In short: Project 19 = (context-tree KV objects) + (persistent cache tier) +
(P2 multi-LoRA pool) + (two small routers) + (RL harness-control training),
gated on solving the `blend_by_reanchor` KV-splice kernel.

---

## 8. Minimal Viable Spike Plan

Sequenced so each spike produces a go/no-go with a *continuation-equivalence*
test (per Project 18 discipline), cheapest de-risk first.

**Spike A — Embedding + ANN infra (unblocks everything).** Stand up a second
llama.cpp context with pooling (or model2vec/external) to embed generation spans
and KV-object metadata; build a small ANN index. Deliverable: given a live span,
retrieve top-k KV objects. This is gap #1 from the reactive memo and R_kv's
foundation. No model changes; pure infra.

**Spike B — KV-splice kernel go/no-go (the crux, P1).** Implement
`blend_by_reanchor`: a native ggml shim (or upstream `kv_rope_shift`) that
RoPE-re-anchors a cached KV object to new positions, plus CacheBlend-style
healing of the top-deviation tokens. **Gate:** does re-anchor+heal produce a
continuation token-equivalent (or reward-equivalent) to full re-decode, *and* is
it cheaper on the 3060? If no on cost, fall back to `blend_by_redecode` and
reframe the whole project around partial re-prefill economics. This is the
single most important experiment.

**Spike C — Wire middleware into batched loops (gap #2).** Route the `on_logits`
/ control hooks through `router.py` / `batching.py` / `context_pool_router.py`
so a branch can be biased/spliced mid-decode. Small, mechanical, high-leverage.

**Spike D — Manual harness end-to-end (heuristic routers).** No learning yet:
hard-coded R_kv (retrieve + model2vec rank) and R_task (dispatch a fixed adapter
on a keyword trigger) into the multi-LoRA pool, fold results back. Measure: does
a heuristic splice/dispatch ever beat the unspliced baseline on a real task
(e.g. our tiny-LoRA eval: ner-json / acrouter / brick-complexity with
Qwen3.5-0.8B tiny base)? Establishes the reward oracle for Stage 0.

**Spike E — Control-head warmup (Stage 0).** Add the harness-control head to L,
supervise it on Spike D's oracle-labeled trajectories. Gate: L requests splices/
dispatches with better-than-random precision.

**Spike F — RL over harness actions (Stage 1), small.** Outcome-driven RL on a
narrow task family with the cost-penalized reward. Gate: learned policy beats the
heuristic harness *and* the no-harness baseline on task-reward-per-latency.

Everything before Spike F is plumbing that stands on its own; Spike F is where
the thesis ("a small model trained to drive this harness wins") is actually
tested. Spike B is the gate that decides whether the elegant zero-prefill version
is viable or whether we live in the partial-re-prefill world.

---

## 9. One-paragraph summary

Keep one small strong dense model (Nanbeige/Ornith-class, ideally with
Nanbeige4.2's looped-transformer flavor) in a tight decode loop, and give it a
learned harness that between iterations (a) splices precomputed KV objects from a
tiered database into its attention context and (b) dispatches specialized
side-jobs to a batched multi-LoRA pool, folding results back — with two small
router models choosing splices/dispatches concurrently with decode, and the loop
model trained by outcome-driven RL to use all of this effectively and cheaply.
It is MoE and MoA lifted out of the weights into the inference harness, with a
third expert axis (experts over context/KV) that standard architectures lack.
The whole thing is gated on one hard, already-scoped problem in this repo:
correct, cheap KV splicing across positions (`blend_by_reanchor`).

---

## 10. Sources

- Nanbeige4-3B Technical Report — https://arxiv.org/abs/2512.06266 ; Nanbeige4.1-3B — https://arxiv.org/html/2602.13367v1 ; Nanbeige4.2-3B looped transformer — https://hackernoon.com/inside-nanbeige42-3b-bases-looped-transformer-architecture ; HF org — https://huggingface.co/Nanbeige
- Switch Transformer / Mixtral / DeepSeek-MoE — https://github.com/deepseek-ai/DeepSeek-MoE ; review — https://sh-tsang.medium.com/review-deepseekmoe-towards-ultimate-expert-specialization-in-mixture-of-experts-language-models-e1536c4304cb
- S-LoRA — https://arxiv.org/abs/2311.03285 ; MLSys PDF — https://proceedings.mlsys.org/paper_files/paper/2024/file/906419cd502575b617cc489a1a696a67-Paper-Conference.pdf ; LMSYS blog (Punica/SGMV) — https://www.lmsys.org/blog/2023-11-15-slora/
- Mixture of LoRA Experts (MoLE) / LD-MoLE / DynMoLE / LoRAHub — https://arxiv.org/abs/2509.25684 ; https://arxiv.org/pdf/2504.00661
- PromptCache — https://arxiv.org/pdf/2311.04934 ; CacheBlend — https://arxiv.org/abs/2405.16444 ; EPIC — https://arxiv.org/pdf/2410.15332 ; RAGCache — https://arxiv.org/pdf/2404.12457
- KVLink — https://arxiv.org/abs/2502.16002 ; CacheClip — https://arxiv.org/abs/2510.10129 ; MiniPIC (IBM/vllm fork) — https://github.com/IBM/vllm ; LMCache (CacheBlend OSS implementation) — https://github.com/LMCache/LMCache
- llama.cpp KV-cache seq-ops source — https://github.com/ggml-org/llama.cpp/blob/master/src/llama-kv-cache.cpp ; issue #5652 (RoPE shift crashes on q4_0 K) — https://github.com/ggml-org/llama.cpp/issues/5652 ; issue #26207 (prompt cache reused across LoRA adapters) — https://github.com/ggml-org/llama.cpp/issues/26207 ; discussion #24944 (shift disabled for M-RoPE) — https://github.com/ggml-org/llama.cpp/discussions/24944 ; discussion #13606 (n_cache_reuse) — https://github.com/ggml-org/llama.cpp/discussions/13606
- MLA — DeepSeek-V2 https://arxiv.org/abs/2405.04434 ; TransMLA https://arxiv.org/abs/2502.07864 ; X-EcoMLA https://arxiv.org/abs/2503.11132 ; MHA2MLA https://arxiv.org/abs/2502.14837 ; MHA2MLA-VLM https://arxiv.org/abs/2601.11464 ; Whisper-MLA https://arxiv.org/abs/2603.00563 ; Kimi K2 https://arxiv.org/abs/2507.20534
- KV-cache quantization — KVQuant https://arxiv.org/abs/2401.18079 ; KIVI https://arxiv.org/abs/2402.02750 ; QuaRot https://arxiv.org/abs/2404.00456
- Nanbeige4.2-3B (LoopSplit / mHC / n-gram preview) — https://huggingface.co/Nanbeige/Nanbeige4.2-3B ; technical report https://arxiv.org/abs/2607.22083
- StreamingLLM / attention sinks — https://arxiv.org/pdf/2309.17453 ; https://github.com/mit-han-lab/streaming-llm
- REST (retrieval-based speculative decoding) — https://arxiv.org/pdf/2311.08252 ; Speculative RAG — https://research.google/blog/speculative-rag-enhancing-retrieval-augmented-generation-through-drafting/
- ReTool (RL for strategic tool use) — https://arxiv.org/abs/2504.11536 ; Offline RL harness control — https://arxiv.org/html/2607.05458
- Universal Transformer (Dehghani et al., 2018) — https://arxiv.org/abs/1807.03819 ; Adaptive Computation Time (Graves, 2016) — https://arxiv.org/abs/1603.08983 ; CoTFormer (Mohtashami et al., 2023) — https://arxiv.org/abs/2310.10845
- Context distillation — Snell et al., "Learning by Distilling Context" (2022) — https://arxiv.org/abs/2209.15189 ; Mu et al., "Learning to Compress Prompts with Gist Tokens" (2023) — https://arxiv.org/abs/2304.08467
