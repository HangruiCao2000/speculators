import logging
import warnings
from copy import deepcopy
from functools import cache, partial

import torch
from torch import nn
from transformers import AutoConfig, PretrainedConfig
from transformers.models.gemma3.modeling_gemma3 import Gemma3RMSNorm
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

from speculators.utils.loading import list_checkpoint_keys

logger = logging.getLogger(__name__)

# Verifier families whose final norm follows the Gemma convention
# `x_norm * (1 + w)` instead of the plain RMSNorm `x_norm * w`. Matched
# exactly against the verifier's effective `model_type` (after text_config
# extraction) — gemma3n and gemma4 dropped this convention, so prefix
# matching would over-include. Qwen3.5, Qwen3-Next and Step3.5 are in here
# because their transformers RMSNorm computes `output * (1.0 + weight)` and
# vLLM uses GemmaRMSNorm for them.
GEMMA_STYLE_FINAL_NORM_MODEL_TYPES = frozenset(
    (
        "gemma",
        "gemma2",
        "gemma3",
        "gemma3_text",
        "qwen3_5",
        "qwen3_5_text",
        "qwen3_5_moe",
        "qwen3_5_moe_text",
        "qwen3_next",
        "recurrent_gemma",
        "step3p5",
    )
)


@cache
def _verifier_model_type(name_or_path: str) -> str | None:
    """The verifier's effective model_type, or None when unresolvable."""
    try:
        verifier_config = get_verifier_config(name_or_path)
    except (KeyError, OSError, ValueError):
        logger.warning(
            "Could not resolve a config for verifier %s; assuming the plain "
            "final-norm convention (x * w).",
            name_or_path,
        )
        return None
    return str(getattr(verifier_config, "model_type", "") or "").lower()


def uses_gemma_style_final_norm(config) -> bool:
    """Whether the verifier's final norm applies gain ``1 + w`` (not ``w``)."""
    verifier = getattr(getattr(config, "speculators_config", None), "verifier", None)
    name_or_path = getattr(verifier, "name_or_path", None)
    if not name_or_path:
        return False
    model_type = _verifier_model_type(name_or_path)
    return model_type is not None and model_type in GEMMA_STYLE_FINAL_NORM_MODEL_TYPES


# Nested keys under which a multimodal verifier hides its text config.
_TEXT_CONFIG_KEYS = ("text_config", "decoder", "llm_config", "language_config")
_EPS_CONFIG_KEYS = ("layer_norm_eps", "layer_norm_epsilon", "rms_norm_eps")


@cache
def _verifier_text_config_dict(name_or_path: str) -> dict:
    """The verifier's effective text config as a plain dict, or {} when unreadable.

    get_config_dict needs no trust_remote_code, unlike AutoConfig on an auto_map verifier.
    """
    try:
        config_dict, _ = PretrainedConfig.get_config_dict(name_or_path)
    except (KeyError, OSError, ValueError) as err:
        logger.warning(
            "Could not read a config.json for verifier %s (%s): %s.",
            name_or_path,
            type(err).__name__,
            err,
        )
        return {}
    for key in _TEXT_CONFIG_KEYS:
        nested = config_dict.get(key)
        if isinstance(nested, dict):
            return nested
    return config_dict


@cache
def _verifier_final_norm_has_bias(name_or_path: str) -> bool:
    """Whether the verifier ships a bias beside its final-norm weight.

    Read from the checkpoint keys, not a model_type allowlist: a bias is what makes the
    norm a LayerNorm, and nparse ships no layer_norm_eps to key off.
    """
    from speculators.model import VERIFIER_FINAL_NORM_WEIGHT_KEYS, _bias_key

    try:
        keys = set(list_checkpoint_keys(name_or_path))
    except (OSError, ValueError, KeyError) as err:
        logger.warning(
            "Could not list checkpoint keys for verifier %s (%s): %s. Assuming its "
            "final norm is an RMSNorm; if it is a LayerNorm the reconstructed training "
            "targets will be missing its bias.",
            name_or_path,
            type(err).__name__,
            err,
        )
        return False
    return any(_bias_key(key) in keys for key in VERIFIER_FINAL_NORM_WEIGHT_KEYS)


def _verifier_name_or_path(config) -> str | None:
    verifier = getattr(getattr(config, "speculators_config", None), "verifier", None)
    return getattr(verifier, "name_or_path", None) or None


def resolve_verifier_norm_class(config) -> type:
    """The norm class matching the verifier's final norm.

    ``nn.LayerNorm`` when the verifier ships a final-norm bias (mBART/BART/Whisper and
    GPT-2-style layouts), otherwise the RMSNorm variant for its gain convention
    (`x * (1 + w)` for the Gemma, Qwen3.5, Qwen3-Next and Step3.5 families, `x * w`
    otherwise). Getting this wrong silently mis-scales every reconstructed target.
    """
    name_or_path = _verifier_name_or_path(config)
    if name_or_path is not None and _verifier_final_norm_has_bias(name_or_path):
        return nn.LayerNorm
    if uses_gemma_style_final_norm(config):
        return Gemma3RMSNorm
    return Qwen3RMSNorm


def resolve_verifier_norm_eps(config) -> float:
    """The verifier's own final-norm eps, falling back to the draft's."""
    draft_eps = float(config.transformer_layer_config.rms_norm_eps)
    name_or_path = _verifier_name_or_path(config)
    if name_or_path is None:
        return draft_eps
    text_config = _verifier_text_config_dict(name_or_path)
    for key in _EPS_CONFIG_KEYS:
        value = text_config.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    # nparse ships no eps key at all, so fall back to nn.LayerNorm's own default.
    if _verifier_final_norm_has_bias(name_or_path):
        return 1e-5
    return draft_eps


def resolve_verifier_norm_width(config) -> int:
    """The verifier's hidden size -- verifier_norm is applied to ITS hidden states."""
    return int(
        getattr(config, "target_hidden_size", None)
        or config.transformer_layer_config.hidden_size
    )


def conditional_torch_compile(func=None, *args, **kwargs):
    if func is None:
        return partial(conditional_torch_compile, *args, **kwargs)
    if torch.cuda.is_available() and hasattr(torch, "compile"):
        return torch.compile(func, *args, **kwargs)
    return func


def get_verifier_config(
    verifier_name_or_path: str,
    trust_remote_code: bool = False,
) -> PretrainedConfig:
    verifier_config = AutoConfig.from_pretrained(
        verifier_name_or_path,
        trust_remote_code=trust_remote_code,
    )
    if hasattr(verifier_config, "text_config"):
        verifier_config = verifier_config.text_config
    return verifier_config


DEFAULT_TARGET_LAYER_IDS_WARNING = (
    "--target-layer-ids is not explicitly set. Setting target "
    "layers to {target_layer_ids}. If custom target layers were used "
    "when launching vllm datagen, please set them explicitly."
)


def resolve_target_layer_ids(
    target_layer_ids: list[int] | None,
    verifier_name_or_path: str,
    trust_remote_code: bool = False,
) -> list[int]:
    num_layers = get_verifier_config(
        verifier_name_or_path,
        trust_remote_code=trust_remote_code,
    ).num_hidden_layers

    if target_layer_ids is None:
        explicit = False
        target_layer_ids = [2, num_layers // 2, num_layers - 3]
    else:
        explicit = True

    # Layer id ``num_layers`` (the final hidden state) is valid, matching the
    # ids scripts/launch_vllm.py emits with --include-last-layer.
    invalid = (
        not target_layer_ids
        or min(target_layer_ids) < 0
        or max(target_layer_ids) > num_layers
        or len(set(target_layer_ids)) != len(target_layer_ids)
    )
    if invalid:
        if explicit:
            raise ValueError(
                f"target_layer_ids must be distinct and within [0, {num_layers}] "
                f"for a verifier with {num_layers} hidden layers, "
                f"got {target_layer_ids}"
            )
        raise ValueError(
            f"Default target layer ids {target_layer_ids} are invalid for a verifier "
            f"with {num_layers} hidden layers; pass --target-layer-ids explicitly."
        )

    if not explicit:
        warnings.warn(
            DEFAULT_TARGET_LAYER_IDS_WARNING.format(target_layer_ids=target_layer_ids),
            stacklevel=3,
        )
    return target_layer_ids


def flatten_rope_parameters(config: PretrainedConfig) -> PretrainedConfig:
    """Flatten nested per-layer-type ``rope_parameters`` for rotary embedding init.

    Models like Laguna store separate rope configs per layer type
    (``sliding_attention``, ``full_attention``). Rotary embedding classes expect
    a flat dict with ``rope_type``/``rope_theta`` at the top level. This helper
    selects the ``sliding_attention`` variant when nested parameters are detected
    and returns a deep-copied config; otherwise returns the original unchanged.
    """
    rope_params = getattr(config, "rope_parameters", None)
    if not rope_params or "sliding_attention" not in rope_params:
        return config
    config = deepcopy(config)
    config.rope_parameters = rope_params["sliding_attention"]
    return config


def resolve_draft_intermediate_size(verifier_config: PretrainedConfig) -> int:
    """Resolve a dense draft MLP ``intermediate_size`` from a verifier config.

    The draft is an independent small *dense* decoder, so its FFN width is a design
    choice rather than something to reconcile with the verifier's routed capacity:

    * Dense verifiers expose ``intermediate_size`` directly; the draft mirrors it.
    * MoE verifiers have no dense ``intermediate_size`` (their FFN is a routed set of
      small experts), so the draft falls back to the widely used ``3 * hidden_size``
      gated-MLP ratio -- the Qwen3 dense convention that the dflash draft decoder
      follows. Pass ``--draft-config`` to set it explicitly instead.

    :raises ValueError: when the verifier config exposes neither ``intermediate_size``
        nor ``hidden_size`` (degenerate config; pass ``--draft-config``).
    """
    dense = getattr(verifier_config, "intermediate_size", None)
    if dense is not None:
        return int(dense)

    hidden_size = getattr(verifier_config, "hidden_size", None)
    if hidden_size is None:
        raise ValueError(
            "Verifier config exposes neither `intermediate_size` nor `hidden_size`, "
            "so a draft intermediate_size cannot be inferred. Pass --draft-config to "
            "set the draft architecture explicitly."
        )

    intermediate_size = 3 * int(hidden_size)
    warnings.warn(
        "Verifier config has no dense intermediate_size (likely MoE); using draft "
        f"intermediate_size={intermediate_size} (3 x hidden_size = {hidden_size}). "
        "Pass --draft-config to override.",
        stacklevel=3,
    )
    return intermediate_size
