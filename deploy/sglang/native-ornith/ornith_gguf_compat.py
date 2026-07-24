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
    """Register a servable text-only model under the name GGUF rewrites to.

    ``sglang.srt.utils.hf_transformers.config.get_config`` forcibly rewrites
    a GGUF checkpoint's ``config.architectures`` to Transformers'
    ``MODEL_FOR_CAUSAL_LM_MAPPING_NAMES[model_type]`` entry -- for
    ``model_type="qwen3_5"`` that is ``"Qwen3_5ForCausalLM"``, unconditionally,
    for every GGUF load regardless of what architecture the source config
    declared. But ``sglang.srt.models.qwen3_5``'s ``EntryClass`` list only
    registers the multimodal ``Qwen3_5ForConditionalGeneration`` /
    ``Qwen3_5MoeForConditionalGeneration`` wrappers, and the plain
    ``Qwen3_5ForCausalLM`` class defined in the same file -- despite its name
    -- is **not** a servable causal LM: it is the flat decoder stack
    (``embed_tokens`` / ``layers`` / ``norm``), its ``params_dict`` has no
    ``model.`` prefix and no ``lm_head``, and its ``forward`` returns raw
    hidden states, not logits. Registering that flat class (as an earlier
    revision of this spike did) makes every GGUF tensor -- whose names are
    ``model.``-prefixed and include ``lm_head`` -- miss ``params_dict``, and
    leaves nothing to turn hidden states into tokens.

    Register instead the text-only ``OrnithTextForCausalLM`` wrapper, which
    owns the flat decoder as ``self.model`` plus an ``lm_head`` and a
    ``LogitsProcessor`` -- a param tree that matches the GGUF-derived names --
    without pulling in the vision tower the ``ConditionalGeneration`` wrappers
    build. See ``ornith_text_model`` for the full rationale.
    """
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ornith_text_model import OrnithTextForCausalLM

    from sglang.srt.models.registry import ModelRegistry

    # Force our complete wrapper even if a prior import registered the flat
    # decoder under this name.
    ModelRegistry.models["Qwen3_5ForCausalLM"] = OrnithTextForCausalLM


def install_qwen3_5_text_config_hybrid_properties() -> None:
    """Graft SGLang's hybrid-GDN config properties onto the Transformers config.

    SGLang sizes and allocates the Gated-DeltaNet SSM state cache from
    ``config.mamba2_cache_params`` and partitions layers with
    ``config.linear_layer_ids`` / ``full_attention_layer_ids`` -- properties
    defined on SGLang's *own* ``Qwen3NextConfig`` (the base of SGLang's
    ``Qwen3_5TextConfig``). But this Transformers version natively owns
    ``model_type="qwen3_5_text"``, so ``AutoConfig`` resolves the resolved
    config to *Transformers'* identically-named ``Qwen3_5TextConfig``, which
    lacks those properties -- and ``handle_max_mamba_cache`` then fails with
    ``'Qwen3_5TextConfig' object has no attribute 'mamba2_cache_params'``.

    Each of these properties reads only fields the resolved config already
    carries (``layers_block_type`` and the ``linear_*`` head/dim/conv sizes),
    so copy the descriptors across verbatim rather than reimplementing them.
    ``HybridLayerType.full_attention == "attention"`` -- exactly the value
    ``resolve_ornith_gguf_config.py`` writes into ``layers_block_type`` -- so
    the layer partitioning lines up with no further translation.
    """
    from transformers.models.qwen3_5.configuration_qwen3_5 import (
        Qwen3_5TextConfig as TransformersQwen3_5TextConfig,
    )

    from sglang.srt.configs.qwen3_next import Qwen3NextConfig

    for name in (
        "mamba2_cache_params",
        "linear_layer_ids",
        "full_attention_layer_ids",
    ):
        if not hasattr(TransformersQwen3_5TextConfig, name):
            setattr(TransformersQwen3_5TextConfig, name, getattr(Qwen3NextConfig, name))


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


def install_gdn_ssm_tensor_map_fix() -> None:
    """Add the two Gated-DeltaNet tensors SGLang's GGUF map builder drops.

    ``GGUFModelLoader._get_gguf_weights_map`` builds its gguf->hf name map by
    walking a dummy HF model's ``state_dict`` and, for each parameter,
    splitting off the final dotted component as a ``weight``/``bias`` suffix
    (``name, suffix = hf_name.rsplit(".", 1)``) before reverse-looking-up the
    remaining ``name`` in the ``gguf`` package's tensor-name map. That shape
    assumption holds for ``nn.Linear`` weights but breaks for the two bare
    ``nn.Parameter`` tensors in ``Qwen3_5GatedDeltaNet``:

    - ``linear_attn.A_log`` -- the ``gguf`` map *does* know this HF name
      (``model.layers.{bid}.linear_attn.A_log`` -> ``blk.{bid}.ssm_a``), but
      SGLang's ``rsplit`` mangles it to ``model.layers.{bid}.linear_attn``
      (treating ``A_log`` as the suffix), which maps to nothing.
    - ``linear_attn.dt_bias`` -- genuinely absent from the ``gguf`` package's
      ``qwen35`` tensor map (its ``SSM_DT`` HF patterns cover ``dt_proj``/``dt``
      only), even though the GGUF file ships the tensor as
      ``blk.{bid}.ssm_dt.bias``.

    Both are per-linear-attention-layer. Without them, 48 of the 427 GGUF
    tensors (24 layers x {A_log, dt_bias}) are silently dropped and left at
    their randomly-initialized values -- exactly the class of GDN weight-load
    gap the earlier runtime evidence flagged. This adds the two missing
    ``gguf_name -> hf_name`` entries for each linear-attention layer, using the
    resolved config's ``layers_block_type`` to target only the layers that
    actually have a Gated-DeltaNet mixer. The HF names it emits
    (``...linear_attn.A_log`` / ``...linear_attn.dt_bias``) are exactly the
    parameter names ``Qwen3_5ForCausalLM.load_weights`` places directly into
    ``params_dict`` (no stacked-mapping rewrite), matching the SGLang model's
    own ``nn.Parameter`` names.
    """
    from sglang.srt.model_loader.loader import GGUFModelLoader

    _original = GGUFModelLoader._get_gguf_weights_map

    def _get_gguf_weights_map_with_gdn_ssm(self, model_config):
        gguf_to_hf_name_map = _original(self, model_config)

        config = model_config.hf_config
        block_types = getattr(config, "layers_block_type", None) or []
        num_layers = getattr(config, "num_hidden_layers", len(block_types))
        for layer_id in range(num_layers):
            # Only linear-attention layers carry a Gated-DeltaNet mixer; the
            # periodic full-attention layers have neither tensor. Default to
            # treating a layer as linear when block metadata is unavailable so
            # the (harmless, file-absent) key simply goes unused.
            block_type = block_types[layer_id] if layer_id < len(block_types) else "linear_attention"
            if block_type not in ("linear_attention", "linear"):
                continue
            gguf_to_hf_name_map.setdefault(
                f"blk.{layer_id}.ssm_a",
                f"model.layers.{layer_id}.linear_attn.A_log",
            )
            gguf_to_hf_name_map.setdefault(
                f"blk.{layer_id}.ssm_dt.bias",
                f"model.layers.{layer_id}.linear_attn.dt_bias",
            )
        return gguf_to_hf_name_map

    GGUFModelLoader._get_gguf_weights_map = _get_gguf_weights_map_with_gdn_ssm


def install_gdn_packed_gguf_loader_binding() -> None:
    """Let the GDN fused-projection packed loader drive GGUF-quant params.

    ``Qwen3_5GatedDeltaNet`` merges the checkpoint's split
    ``in_proj_qkv``/``in_proj_z`` (and ``in_proj_b``/``in_proj_a``) into single
    ``MergedColumnParallelLinear`` modules and, in ``load_weights``, loads the
    ``in_proj_qkv`` shard with a *tuple* ``loaded_shard_id`` ``(0, 1, 2)`` (it
    itself packs q+k+v). The base ``LinearBase.weight_loader`` explicitly
    rejects tuple shard ids (``NotImplementedError: Shard id with multiple
    indices ... use weight_loader_v2``); the model works around this by
    rebinding a packed loader that splits the tuple into per-output-block int
    shards. But ``_bind_packed_weight_loaders`` only rebinds the ``weight`` /
    ``*_scale`` params -- under ``--load-format gguf`` the merged linear's real
    params are ``qweight`` / ``qweight_type`` instead, so the packed loader
    never gets installed and the tuple-shard load crashes.

    Extend the rebinding to cover the GGUF params. Splitting the fused
    projection along its output dimension (whole rows) is safe for llama.cpp
    block quant, which quantizes along the *input* dimension within each row --
    so each output row's packed bytes stay intact -- and each resulting int
    shard then flows through the same GGUF ``MergedColumnParallelLinear`` path
    that already loads e.g. ``gate_up_proj`` from split GGUF tensors.
    """
    from sglang.srt.models.qwen3_5 import Qwen3_5GatedDeltaNet

    _original = Qwen3_5GatedDeltaNet._bind_packed_weight_loaders

    def _make_gguf_type_packed_loader(original_weight_loader):
        """Fan a split tensor's single GGUF quant-type scalar out to its shards.

        ``qweight_type`` is a per-shard 0-d code array; the base GGUF loader
        does ``param.data[idx].copy_(loaded_weight)`` with a 0-d
        ``loaded_weight``. The generic packed loader would ``view(-1)`` that
        scalar to shape ``[1]`` and mismatch the 0-d destination, so route the
        type scalar straight to each int shard unchanged (all sub-blocks of a
        single split checkpoint tensor share one quant type).
        """

        def weight_loader(param, loaded_weight, loaded_shard_id=None):
            if isinstance(loaded_shard_id, tuple):
                for idx in loaded_shard_id:
                    original_weight_loader(param, loaded_weight, idx)
                return
            return original_weight_loader(param, loaded_weight, loaded_shard_id)

        return weight_loader

    def _bind_packed_weight_loaders_with_gguf(self, module):
        _original(self, module)
        # Packed (output-dim split) loader for the quantized weight bytes.
        qweight = getattr(module, "qweight", None)
        if qweight is not None:
            original_loader = getattr(qweight, "weight_loader", None)
            if original_loader is not None:
                self._override_weight_loader(
                    qweight, self._make_packed_weight_loader(module, original_loader)
                )
        # Scalar-broadcast loader for the per-shard quant-type codes.
        qweight_type = getattr(module, "qweight_type", None)
        if qweight_type is not None:
            original_loader = getattr(qweight_type, "weight_loader", None)
            if original_loader is not None:
                self._override_weight_loader(
                    qweight_type, _make_gguf_type_packed_loader(original_loader)
                )

    Qwen3_5GatedDeltaNet._bind_packed_weight_loaders = (
        _bind_packed_weight_loaders_with_gguf
    )


def install_gdn_ba_unquantized() -> None:
    """Build the tiny GDN ``in_proj_ba`` projection unquantized.

    This ``UD-Q4_K_XL`` GGUF keeps the low-rank beta/alpha projections
    (``ssm_beta`` / ``ssm_alpha`` -> ``in_proj_b`` / ``in_proj_a``, each only
    ``num_v_heads`` outputs) in **F32**, not block-quantized -- unsloth's
    dynamic quant leaves such small, precision-sensitive tensors full-width.
    SGLang builds ``in_proj_ba`` with the layer's quant config regardless, so
    the merged param is ``qweight``/``qweight_type`` while the checkpoint only
    provides plain ``.weight`` tensors, and the load misses. Because the GGUF
    representation is already F32 here, build ``in_proj_ba`` with no quant
    config so the merged param is a normal ``.weight`` -- which the existing
    ``_bind_packed_weight_loaders`` (``weight`` branch) already fans the split
    ``in_proj_b`` / ``in_proj_a`` shards into.
    """
    from sglang.srt.models.qwen3_5 import Qwen3_5GatedDeltaNet

    _original = Qwen3_5GatedDeltaNet.create_ba_proj

    def create_ba_proj_unquantized(
        self, hidden_size, num_v_heads, quant_config, prefix, tp_rank=None, tp_size=None
    ):
        return _original(
            self, hidden_size, num_v_heads, None, prefix, tp_rank, tp_size
        )

    Qwen3_5GatedDeltaNet.create_ba_proj = create_ba_proj_unquantized


def install_hybrid_gdn_config_recognition() -> None:
    """Make SGLang recognize the resolved text config as a hybrid-GDN model.

    ``ModelRunner.hybrid_gdn_config`` gates whether the runner builds a
    ``HybridLinearAttnBackend`` (routing the Gated-DeltaNet layers to the
    linear-attention/mamba backend and the periodic full-attention layers to
    flashinfer) plus the SSM state cache. It does so with
    ``isinstance(hf_config.get_text_config(), Qwen3NextConfig | Qwen3_5Config |
    ...)``. This spike deliberately publishes the flat ``Qwen3_5TextConfig``
    (``model_type="qwen3_5_text"``) rather than the multimodal ``Qwen3_5Config``
    wrapper, so the isinstance check misses, ``mambaish_config`` is ``None``,
    and **no** linear-attention backend is created. At runtime the GDN layers
    then call ``get_attn_backend()`` and get the full-attention backend, which
    raises ``AttentionBackend.forward() missing ... 'q', 'k', 'v'``.

    The text config carries every field the hybrid path reads
    (``linear_key_head_dim`` / ``linear_num_value_heads`` /
    ``layers_block_type`` / ...), so return it directly for our known
    text model_types when the built-in check declines.
    """
    from sglang.srt.model_executor.model_runner import ModelRunner

    _original = ModelRunner.hybrid_gdn_config.fget

    def _hybrid_gdn_config(self):
        result = _original(self)
        if result is not None:
            return result
        text_config = self.model_config.hf_config.get_text_config()
        if getattr(text_config, "model_type", None) in ("qwen3_5", "qwen3_5_text"):
            return text_config
        return None

    ModelRunner.hybrid_gdn_config = property(_hybrid_gdn_config)
