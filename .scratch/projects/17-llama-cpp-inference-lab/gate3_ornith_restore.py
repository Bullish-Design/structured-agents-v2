"""Gate 3: does Ornith-1.0-9B hybrid (GatedDeltaNet) state survive save/restore?

Deterministic greedy continuation must be IDENTICAL after restoring the
post-prompt state. Two scenarios:
  A) same-instance   save_state -> load_state   (state serialize/restore correct?)
  B) cross-instance  save_state -> NEW Llama.load_state  (restart scenario the
     cache pillar actually needs)

If hybrid recurrent state is dropped/mishandled, restored continuation diverges.
"""
import numpy as np
from llama_cpp import Llama

MODEL = "/home/andrew/.cache/structured-agents/models/Ornith-1.0-9B-UD-Q4_K_XL.gguf"
N_GEN = 16
PROMPT = b"The quick brown fox jumps over the lazy dog. In summary, the sentence describes"

def mk():
    return Llama(model_path=MODEL, n_ctx=1024, n_batch=256, n_threads=8,
                 logits_all=False, verbose=False)

def greedy(llm, n):
    out = []
    for _ in range(n):
        logits = np.ctypeslib.as_array(llm._ctx.get_logits(), shape=(llm._n_vocab,))
        tid = int(np.argmax(logits))
        out.append(tid)
        llm.eval([tid])
    return out

llm = mk()
toks = llm.tokenize(PROMPT)
prefix, last = toks[:-1], toks[-1]
print(f"[prompt] {len(toks)} tokens; checkpoint after prefix={len(prefix)}, suffix decodes token {last}")

# NOTE: state save does NOT include the output logits buffer — restoring the
# prefix KV leaves no fresh logits. So the correct continuation (and the real
# prefix-cache pattern) restores the prefix then DECODES the suffix; the first
# real logits come from that suffix decode. Both baseline and restored paths
# therefore eval([last]) from the same restored position → fair comparison.

# --- baseline ---
llm.reset(); llm.eval(prefix)
state = llm.save_state()
print(f"[state] n_tokens={state.n_tokens}  llama_state_size={state.llama_state_size} bytes")
llm.eval([last]); baseline = greedy(llm, N_GEN)
print(f"[baseline] {baseline}")

# --- A) same-instance restore ---
llm.load_state(state)
llm.eval([last]); restored_a = greedy(llm, N_GEN)
match_a = restored_a == baseline
print(f"[A same-instance] match={match_a}")
if not match_a:
    print(f"   restored={restored_a}")

# --- B) cross-instance restore (restart scenario) ---
llm2 = mk()
llm2.load_state(state)
llm2.eval([last]); restored_b = greedy(llm2, N_GEN)
match_b = restored_b == baseline
print(f"[B cross-instance] match={match_b}")
if not match_b:
    print(f"   restored={restored_b}")

print(f"\n[GATE 3] same-instance={'PASS' if match_a else 'FAIL'}  "
      f"cross-instance={'PASS' if match_b else 'FAIL'}")
