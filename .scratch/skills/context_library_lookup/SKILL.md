---
name: context_library_lookup
description: >
  Locate and extract authoritative documentation and source code for the
  vendored xgrammar and vLLM materials stored under the repository's .context/
  directory.
---

# SKILL: Lookup xgrammar + vLLM context from `.context/`

## Purpose

Ground answers about **xgrammar** and **vLLM** in the **exact vendored
materials** in this repo. The `.context/` directory contains documentation,
examples, and (for xgrammar) the Python package source. Use it instead of
upstream web docs to avoid version mismatch.

## When to Use

Use this skill whenever questions involve:

- Grammar-constrained decoding, EBNF/regex grammars, or XGrammar APIs.
- vLLM structured outputs, tool calling, FunctionGemma integration, or server behavior.
- Any symbols likely defined in xgrammar or vLLM rather than this repo.

Do NOT use external web docs if the relevant info exists in `.context/`.

## Vendored Roots (Actual Layout)

Top-level context docs:

- `.context/CUSTOM_XGRAMMAR_GUIDE.md`
- `.context/FUNCTIONGEMMA_DOCS.md`
- `.context/FUNCTIONGEMMA_PROMPT_TIPS.md`
- `.context/STRUCTURAL_TAG_INTEGRATION_EXPLORATION.md`
- `.context/SERVER_SETUP.md`
- `.context/SERVER_LOG_GUIDE.md`
- `.context/HOW_TO_CREATE_AN_AGENT.md`
- `.context/HOW_TO_CREATE_A_GRAIL_PYM_SCRIPT.md`
- `.context/functiongemma_examples/`

xgrammar materials:

- `.context/xgrammar/README.md`
- `.context/xgrammar/docs/`
- `.context/xgrammar/examples/`
- `.context/xgrammar/python/` (Python package source)

vLLM materials:

- `.context/vllm/README(16).md`
- `.context/vllm/tool_calling.md`
- `.context/vllm/structured_outputs.md`
- `.context/vllm/VLLM_FUNCTIONGEMMA_DOCS.md`
- `.context/vllm/*.py` (examples and scripts)
- `.context/vllm/lora.md`

## Core Workflow

1. Identify whether the concept is **xgrammar** or **vLLM**.
2. Start with docs/README files, then check examples.
3. For xgrammar APIs, use `.context/xgrammar/python/` for canonical source.
4. Extract minimal authoritative excerpts (signature + key logic/notes).
5. Answer with file paths + line references.

## Search Tactics (ripgrep)

Start narrow:

xgrammar:

    rg -n "MySymbol" .context/xgrammar
    rg -n "EBNF|grammar|constraint" .context/xgrammar
    rg -n "def my_function" .context/xgrammar/python

vLLM:

    rg -n "tool calling|structured outputs" .context/vllm
    rg -n "FunctionGemma" .context/vllm
    rg -n "server|openai" .context/vllm

If origin is unclear, search both:

    rg -n "SomeUniqueString" .context/xgrammar .context/vllm

## Documentation First

Priority order:

1. `.context/*` topic guides (FunctionGemma, server setup, structural tags)
2. Library READMEs and docs directories
3. Example scripts
4. xgrammar Python source (canonical implementation)

Note: The vLLM export here is documentation + example scripts, not full source.

## Canonical Definition Rules

### xgrammar

- Prefer `.context/xgrammar/python/` for API definitions and logic.
- Use docs to explain intended usage or constraints.

### vLLM

- Prefer `.context/vllm/*.md` for behavior and configuration.
- Use example scripts to clarify expected request shapes or usage patterns.

## Extract Minimal Authoritative Excerpts

Include:

- Function/class signature or relevant section header
- Docstring or paragraph that defines semantics
- The minimal snippet that answers the question

Avoid dumping whole files.

## Response Template

1. **Library**: xgrammar or vLLM
2. **Where in `.context/`**: root + key file(s)
3. **Docs say**: short summary with file reference
4. **Implementation** (if xgrammar): key snippet with file reference
5. **Behavior summary** tied to cited lines

## Anti-Patterns

- Don’t rely on upstream GitHub docs if `.context/` has it.
- Don’t assume behavior from memory.
- Don’t cite only index/re-export files.
- Don’t paste huge code blocks.

## Quick Cheatsheet

### Grammar-constrained decoding
- Start: `.context/CUSTOM_XGRAMMAR_GUIDE.md`, `.context/xgrammar/README.md`
- Then: `.context/xgrammar/docs/`, `.context/xgrammar/python/`

### vLLM structured outputs/tool calling
- Start: `.context/vllm/structured_outputs.md`, `.context/vllm/tool_calling.md`
- Then: `.context/vllm/VLLM_FUNCTIONGEMMA_DOCS.md`, example scripts

### FunctionGemma integration
- Start: `.context/FUNCTIONGEMMA_DOCS.md`, `.context/FUNCTIONGEMMA_PROMPT_TIPS.md`
- Then: `.context/functiongemma_examples/`, `.context/vllm/VLLM_FUNCTIONGEMMA_DOCS.md`

## Goal

Produce answers that are **verifiably grounded** in the vendored xgrammar and
vLLM materials in `.context/`, with file/line references for quick confirmation.
