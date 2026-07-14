# Structured Agents Gemma 4 fork

This directory vendors `vllm-project/vllm-gguf-plugin` release `v0.0.4` at
commit `a1746015d6ab9a8db63391b62dfb6089213b86fb`.

The local patch adds only the native vLLM Gemma 4 GGUF adapter required by
Structured Agents. It is based on the mapping direction in vLLM PRs #40281
and #41589, adapted to the out-of-tree plugin API and `vllm==0.25.0`.
The MoE handled-name regression noted during review is fixed by retaining the
`.weight` suffix in `build_gemma4_moe_tensor_map`.
