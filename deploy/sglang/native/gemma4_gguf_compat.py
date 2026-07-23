"""Temporary Gemma 4 GGUF conversion compatibility layer.

Removal condition: delete this module and its sitecustomize installation once
the pinned Transformers revision converts Gemma 4 GGUF mixed-attention metadata
to a valid Gemma4TextConfig, permits the resulting SGLang Gemma 4 constructor
without an explicit eager override, and SGLang handles ``gemma4_text``
equivalently to the text configuration nested in native Gemma 4 checkpoints.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def _field(reader: Any, name: str, default: Any = None) -> Any:
    field = reader.fields.get(name)
    return default if field is None else field.contents()


def _convert_gemma4_config(gguf_path: str, parsed: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct native Gemma 4 text config semantics from GGUF metadata."""
    config = parsed.get("config", {})
    if config.get("model_type") != "gemma4_text":
        return parsed

    from gguf import GGUFReader

    reader = GGUFReader(gguf_path)
    architecture = _field(reader, "general.architecture")
    if architecture != "gemma4":
        return parsed

    pattern = _field(reader, "gemma4.attention.sliding_window_pattern")
    kv_heads = _field(reader, "gemma4.attention.head_count_kv")
    if not isinstance(pattern, list) or not isinstance(kv_heads, list):
        raise ValueError("Gemma 4 GGUF requires per-layer attention metadata")
    if len(pattern) != len(kv_heads) or len(pattern) != config["num_hidden_layers"]:
        raise ValueError("Gemma 4 GGUF per-layer attention metadata has inconsistent length")

    sliding_indices = [i for i, is_sliding in enumerate(pattern) if is_sliding]
    full_indices = [i for i, is_sliding in enumerate(pattern) if not is_sliding]
    if not sliding_indices or not full_indices:
        raise ValueError("Gemma 4 GGUF must include both sliding and full-attention layers")

    sliding_kv = {kv_heads[i] for i in sliding_indices}
    full_kv = {kv_heads[i] for i in full_indices}
    if len(sliding_kv) != 1 or len(full_kv) != 1:
        raise ValueError("Gemma 4 GGUF has heterogeneous KV heads within one attention class")

    # Full-attention Gemma 4 layers have no V projection when K is shared with V.
    missing_full_v = all(f"blk.{i}.attn_v.weight" not in {t.name for t in reader.tensors} for i in full_indices)
    if not missing_full_v:
        raise ValueError("Gemma 4 GGUF full-attention layers unexpectedly contain V projections")

    config.update(
        num_key_value_heads=sliding_kv.pop(),
        num_global_key_value_heads=full_kv.pop(),
        layer_types=["sliding_attention" if is_sliding else "full_attention" for is_sliding in pattern],
        hidden_size_per_layer_input=_field(reader, "gemma4.embedding_length_per_layer_input", 0),
        attention_k_eq_v=True,
        num_kv_shared_layers=_field(reader, "gemma4.attention.shared_kv_layers", 0),
        final_logit_softcapping=_field(reader, "gemma4.final_logit_softcapping"),
    )
    return parsed


def install_transformers_gguf_patch() -> None:
    """Patch both call sites that retain Transformers' GGUF loader reference."""
    import transformers.configuration_utils as configuration_utils
    import transformers.modeling_gguf_pytorch_utils as gguf_utils

    original = gguf_utils.load_gguf_checkpoint
    if getattr(original, "_structured_agents_gemma4_compat", False):
        return

    def patched(gguf_checkpoint_path: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
        parsed = original(gguf_checkpoint_path, *args, **kwargs)
        return _convert_gemma4_config(gguf_checkpoint_path, parsed)

    patched._structured_agents_gemma4_compat = True  # type: ignore[attr-defined]
    gguf_utils.load_gguf_checkpoint = patched
    configuration_utils.load_gguf_checkpoint = patched


def install_gguf_model_type_alias() -> None:
    """Let SGLang's generic GGUF loader find the Gemma 4 text tensor map.

    gguf-py names this architecture ``gemma4``; Transformers names the
    text-only configuration ``gemma4_text``. Both use the same tensor map.
    """
    import gguf

    for arch, model_type in gguf.MODEL_ARCH_NAMES.items():
        if model_type == "gemma4":
            gguf.MODEL_ARCH_NAMES[arch] = "gemma4_text"
            return
    raise RuntimeError("Installed gguf package has no gemma4 architecture")


def normalize_sglang_gemma4_text_config(config: Any) -> Any:
    """Apply SGLang's native Gemma 4 text-config convention to GGUF text configs."""
    if getattr(config, "model_type", None) != "gemma4_text":
        return config
    global_kv_heads = getattr(config, "num_global_key_value_heads", None)
    global_head_dim = getattr(config, "global_head_dim", None)
    if global_kv_heads is None or global_head_dim is None:
        raise ValueError("Gemma 4 GGUF config lacks global attention dimensions")

    config.swa_head_dim = config.head_dim
    config.swa_v_head_dim = config.head_dim
    config.swa_num_key_value_heads = config.num_key_value_heads
    config.head_dim = global_head_dim
    config.num_key_value_heads = global_kv_heads
    config.v_head_dim = config.head_dim
    return config


def prepare_sglang_gemma4_construction(config: Any) -> Any:
    """Use HF eager attention only while SGLang constructs its Gemma 4 module.

    SGLang's Gemma 4 implementation owns the actual runtime attention through
    ``RadixAttention``.  Transformers nevertheless validates its own attention
    selection in ``PreTrainedModel.__init__``; the current pinned configuration
    retains an explicit ``sdpa`` setting, which Gemma 4 does not support there.
    """
    if getattr(config, "model_type", None) == "gemma4_text":
        config._attn_implementation = "eager"
        # Transformers' PreTrainedModel.__init__ now also validates
        # `experts_implementation` unconditionally, defaulting to "grouped_mm"
        # and raising for any class that doesn't support it. Gemma 4's dense
        # (non-MoE) text model has no experts, so pin the only implementation
        # it does support before construction, mirroring the attention override
        # above.
        config._experts_implementation = "eager"
    return config
