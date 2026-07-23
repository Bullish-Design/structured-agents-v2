"""Config-resolution adapter for the SGLang Ornith-1.0-9B GGUF spike.

``transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint`` raises
``ValueError: GGUF model with architecture qwen35 is not supported yet.``
before SGLang can even build a config -- the GGUF metadata parser's
architecture allowlist (``GGUF_CONFIG_MAPPING``) has no ``qwen35`` entry.
This is not an SGLang bug and there is no live-GGUF-derive path to patch
around, unlike Gemma 4 (whose GGUF metadata parser did work, just produced a
config SGLang couldn't build attention layers from correctly).

Instead this uses the real ``config.json`` published in the non-GGUF
``unsloth/Ornith-1.0-9B`` repo directly, resolved once offline by
``resolve_ornith_gguf_config.py``, and redirects SGLang's config parser to
load that static file instead of touching the GGUF's embedded metadata at
all. Weight *tensors* still load through SGLang's ``--load-format gguf``
path -- this only replaces config resolution.
"""

from __future__ import annotations

import os


def install_gguf_model_type_alias() -> None:
    """Make duplicate AutoConfig registration idempotent.

    Newer Transformers already owns a small number of model-type names that
    SGLang 0.5.14 registers at import time. Keep the upstream registration
    when it exists.
    """
    from transformers import AutoConfig

    _original_register = AutoConfig.register

    def _register_with_existing_ok(*args: object, **kwargs: object) -> None:
        kwargs.setdefault("exist_ok", True)
        _original_register(*args, **kwargs)

    AutoConfig.register = _register_with_existing_ok


def install_static_config_redirect(config_dir: str) -> None:
    """Redirect every GGUF-path AutoConfig resolution to a static config.json.

    SGLang resolves the model config from more than one call site --
    ``HfModelConfigParser.parse`` (reached via ``gguf_file=...``) for the
    main engine config, and ``sglang.srt.utils.hf_transformers.processor
    .get_processor``'s direct ``AutoConfig.from_pretrained(model_path, ...)``
    call (no ``gguf_file`` kwarg at all -- it just treats the raw GGUF path
    as a repo id, which Transformers' JSON loader then rejects outright).
    Patching ``AutoConfig.from_pretrained`` itself, rather than either
    individual caller, covers both without hunting down every future one.
    """
    from transformers import AutoConfig

    _original_from_pretrained = AutoConfig.from_pretrained.__func__

    def _from_pretrained_with_ornith_gguf_compat(cls, pretrained_model_name_or_path, *args, **kwargs):
        is_gguf_path = (
            isinstance(pretrained_model_name_or_path, str)
            and pretrained_model_name_or_path.endswith(".gguf")
        ) or "gguf_file" in kwargs
        if is_gguf_path:
            kwargs.pop("gguf_file", None)
            return _original_from_pretrained(cls, config_dir, *args, **kwargs)
        return _original_from_pretrained(
            cls, pretrained_model_name_or_path, *args, **kwargs
        )

    AutoConfig.from_pretrained = classmethod(_from_pretrained_with_ornith_gguf_compat)


def install_causal_lm_registry_entry() -> None:
    """Register Qwen3_5ForCausalLM under SGLang's model registry.

    ``sglang.srt.utils.hf_transformers.config.get_config`` forcibly rewrites
    a GGUF checkpoint's ``config.architectures`` to Transformers'
    ``MODEL_FOR_CAUSAL_LM_MAPPING_NAMES[model_type]`` entry -- for
    ``model_type="qwen3_5"`` that is ``"Qwen3_5ForCausalLM"``, unconditionally,
    for every GGUF load regardless of what architecture the source config
    declared. But ``sglang.srt.models.qwen3_5``'s ``EntryClass`` list only
    registers ``Qwen3_5ForConditionalGeneration`` / ``Qwen3_5MoeForConditionalGeneration``
    -- the plain ``Qwen3_5ForCausalLM`` class it also defines is never
    registered, even though it is a complete, working model implementation.
    This is a gap in SGLang's registry, not a missing architecture; add the
    one missing entry directly rather than waiting on an upstream release.
    """
    from sglang.srt.models.qwen3_5 import Qwen3_5ForCausalLM
    from sglang.srt.models.registry import ModelRegistry

    ModelRegistry.models.setdefault("Qwen3_5ForCausalLM", Qwen3_5ForCausalLM)


def install_text_only_mm_processor_skip() -> None:
    """Skip multimodal-processor lookup for the GGUF-rewritten text arch.

    ``ModelConfig`` auto-enables multimodal support unless the (post-GGUF-
    rewrite) architecture name is on a small hardcoded disabled list that
    does not include ``Qwen3_5ForCausalLM``, and there is no CLI flag to
    force it off (``--enable-multimodal`` is `store_true`-only). This spike
    never loads the vision tower (its GGUF has no vision weights; the
    separate mmproj file is explicitly refused in serve.sh), so there is
    never a real processor to find. Rather than fail startup, make the
    lookup for this one known text-only architecture a deliberate no-op.
    """
    from sglang.srt.managers import multimodal_processor as _mm

    _original_get_mm_processor = _mm.get_mm_processor

    def _get_mm_processor_text_only_skip(hf_config, *args, **kwargs):
        if list(getattr(hf_config, "architectures", []) or []) == ["Qwen3_5ForCausalLM"]:
            return None
        return _original_get_mm_processor(hf_config, *args, **kwargs)

    _mm.get_mm_processor = _get_mm_processor_text_only_skip

    from sglang.srt.managers import tokenizer_manager as _tm

    _tm.get_mm_processor = _get_mm_processor_text_only_skip


def install_dense_expert_location_skip() -> None:
    """Make expert-location bookkeeping a no-op for the dense text model.

    ``Qwen3_5ForCausalLM.get_model_config_for_expert_location`` is shared by
    the dense and MoE subclasses and always returns a
    ``ModelConfigForExpertLocation`` when called, even for a dense model with
    ``num_experts=0``. ``ModelConfigForExpertLocation.from_model_config``
    only skips expert-location setup when the model class has *no* such
    method at all -- so the dense-model 0-experts case falls through to
    ``eplb.expert_location._pad_nested_array``, which calls ``max()`` over an
    empty per-layer expert list and crashes. Returning ``None`` for
    ``num_experts=0`` (this spike's dense checkpoint) is what a model class
    without the method would already produce, and is the intended "no MoE
    expert-location tracking needed" outcome.
    """
    from sglang.srt.models.qwen3_5 import Qwen3_5ForCausalLM

    _original = Qwen3_5ForCausalLM.get_model_config_for_expert_location.__func__

    def _get_model_config_for_expert_location_dense_skip(cls, config):
        text_config = getattr(config, "text_config", config)
        if getattr(text_config, "num_experts", 0) == 0:
            return None
        return _original(cls, config)

    Qwen3_5ForCausalLM.get_model_config_for_expert_location = classmethod(
        _get_model_config_for_expert_location_dense_skip
    )


def install_gguf_arch_name_translation() -> None:
    """Teach SGLang's GGUF tensor-name mapper our resolved config's model_type.

    ``GGUFModelLoader._get_gguf_weights_map`` looks up the tensor-name map by
    searching ``gguf.MODEL_ARCH_NAMES`` (from the standalone ``gguf`` PyPI
    package, not Transformers) for a value equal to ``config.model_type``,
    with a couple of hardcoded translations already in place for mismatches
    between Transformers' and llama.cpp/gguf's naming (``cohere`` ->
    ``command-r``, ``qwen3_moe`` -> ``qwen3moe``). The ``gguf`` package
    already knows the full ``qwen35`` tensor layout, including the
    Gated-DeltaNet linear-attention tensors -- but our resolved config's
    ``model_type`` is Transformers' ``qwen3_5_text``/``qwen3_5``, which has
    no matching entry. Same class of one-line gap as
    ``install_causal_lm_registry_entry``; add the missing translation.
    """
    from sglang.srt.model_loader.loader import GGUFModelLoader

    _original = GGUFModelLoader._get_gguf_weights_map

    def _get_gguf_weights_map_with_translation(self, model_config):
        original_model_type = model_config.hf_config.model_type
        if original_model_type in ("qwen3_5", "qwen3_5_text"):
            model_config.hf_config.model_type = "qwen35"
        try:
            return _original(self, model_config)
        finally:
            model_config.hf_config.model_type = original_model_type

    GGUFModelLoader._get_gguf_weights_map = _get_gguf_weights_map_with_translation
