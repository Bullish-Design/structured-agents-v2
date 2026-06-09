# Verification — output typing & client-side constraint compilation

**Date:** 2026-06-09 · scripts: `spike_output_types.py` (+ inline xgrammar test).
Settles open question #1 and caveats #1/#3 from the first concept draft with facts.

## 1. How PydanticAI 1.87 emits each output type (captured on the wire)

| `output_type` | `response_format` | `tools` / `tool_choice` | Mode |
|---|---|---|---|
| `str` | — | — | **text** (raw content) |
| `Literal[...]` | — | `final_result` / `required` (enum under `response`) | **tool** (JSON) |
| `SomeModel` (plain) | — | `final_result` / `required` | **tool** (JSON) — *this is the default* |
| `NativeOutput(SomeModel)` | `json_schema` | — | **native** (JSON) |

Key takeaways:
- **PydanticAI's default for a Model output is the function-calling "output tool",
  NOT `response_format`.** To get the clean `response_format: json_schema` that vLLM
  XGrammar constrains directly, you must explicitly wrap in `NativeOutput`. → the library
  must apply `NativeOutput` itself; users should never have to remember it.
- **`output_type=str` is true text mode** (no response_format, no tools). This is the
  substrate for bare-string `grammar`/`regex`/`choice` constrained via `extra_body`.
- **`Literal[...]` is accepted but rides the JSON tool path**, not a bare string. So a
  *typed* choice is JSON; a *bare-token* choice needs `str` + `extra_body` choice.
  (Confirms: support both, default routers to the typed json_schema form.)

## 2. Client-side XGrammar compile check is feasible — but heavy

`xgrammar` 0.2.1:
- `Grammar.from_json_schema(json.dumps(Model.model_json_schema()))` → compiles
  **client-side, no server, no tokenizer**. ✅
- `Grammar.from_regex(...)` → compiles. ✅
- A self-recursive model schema → **accepted** (XGrammar handles recursion). ✅

So we *can* validate "is this schema XGrammar-compilable?" at class-definition time and
fail fast. **But** `pip install xgrammar` pulls `torch` + the full CUDA stack (~2 GB,
59 packages). Therefore:
- It must be an **optional dev extra** (e.g. `pip install structured-agents-v2[grammar-check]`),
  never a runtime dependency.
- The check is gated behind a try-import; absent the extra, `ConstrainedOutput` skips it
  (the real enforcement is server-side in vLLM anyway).
