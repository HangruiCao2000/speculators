"""How the verifier's final-norm class, eps and width are resolved.

The class comes from the checkpoint — a bias beside the final-norm weight means the
verifier's norm is a LayerNorm — because a model_type allowlist cannot keep up and some
verifiers (Nemotron-3.5-Parse) ship no eps key to probe either.
"""

from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.utils import (
    _verifier_final_norm_has_bias,
    _verifier_model_type,
    resolve_verifier_norm_class,
    resolve_verifier_norm_eps,
    resolve_verifier_norm_width,
)

from .test_checkpoint_key_ownership import VERIFIER_VOCAB, _make_model


def _write_verifier(tmp_path, name: str, keys: dict, config: dict | None = None):
    """A fake verifier dir with real weight KEYS, so bias detection has ground truth."""
    d = tmp_path / name
    d.mkdir(parents=True)
    (d / "config.json").write_text(
        json.dumps({"model_type": "qwen3", **(config or {})})
    )
    save_file(keys, str(d / "model.safetensors"))
    return str(d)


def _point_at(model, path: str):
    model.config.speculators_config.verifier.name_or_path = path
    _verifier_final_norm_has_bias.cache_clear()
    _verifier_model_type.cache_clear()


def test_layer_norm_verifier_resolves_to_layer_norm(tmp_path):
    """A final-norm bias in the checkpoint means the verifier norm is a LayerNorm."""
    model = _make_model(DFlashDraftModel, VERIFIER_VOCAB)
    _point_at(
        model,
        _write_verifier(
            tmp_path,
            "mbart",
            {
                "decoder.layer_norm.weight": torch.ones(16),
                "decoder.layer_norm.bias": torch.full((16,), 0.5),
                "decoder.layers.0.final_layer_norm.weight": torch.ones(16),
            },
        ),
    )
    assert resolve_verifier_norm_class(model.config) is nn.LayerNorm
    norm = resolve_verifier_norm_class(model.config)(
        resolve_verifier_norm_width(model.config),
        eps=resolve_verifier_norm_eps(model.config),
    )
    assert norm.bias is not None


def test_rms_norm_verifier_is_unchanged(tmp_path):
    """No bias in the checkpoint -> the pre-existing RMSNorm path, same eps."""
    model = _make_model(DFlashDraftModel, VERIFIER_VOCAB)
    _point_at(
        model,
        _write_verifier(
            tmp_path,
            "qwen",
            {
                "model.norm.weight": torch.ones(16),
            },
        ),
    )
    assert resolve_verifier_norm_class(model.config) is Qwen3RMSNorm
    assert resolve_verifier_norm_eps(model.config) == pytest.approx(
        model.config.transformer_layer_config.rms_norm_eps
    )


def test_verifier_eps_wins_over_the_draft_eps(tmp_path):
    """verifier_norm is the VERIFIER's norm, so it takes the verifier's eps."""
    model = _make_model(DFlashDraftModel, VERIFIER_VOCAB)
    _point_at(
        model,
        _write_verifier(
            tmp_path,
            "epsy",
            {"model.norm.weight": torch.ones(16)},
            config={"rms_norm_eps": 1.5e-7},
        ),
    )
    assert resolve_verifier_norm_eps(model.config) == pytest.approx(1.5e-7)


def test_layer_norm_verifier_without_eps_key_uses_layer_norm_default(tmp_path):
    """The Nemotron-3.5-Parse case: a LayerNorm verifier that ships no eps at all."""
    model = _make_model(DFlashDraftModel, VERIFIER_VOCAB)
    _point_at(
        model,
        _write_verifier(
            tmp_path,
            "noeps",
            {
                "decoder.layer_norm.weight": torch.ones(16),
                "decoder.layer_norm.bias": torch.zeros(16) + 0.25,
            },
        ),
    )
    assert resolve_verifier_norm_eps(model.config) == pytest.approx(1e-5)


def test_verifier_norm_width_follows_the_verifier(tmp_path):
    """verifier_norm is applied to the verifier's hidden states, not the draft's."""
    model = _make_model(DFlashDraftModel, VERIFIER_VOCAB)
    assert resolve_verifier_norm_width(model.config) == 16
    model.config.target_hidden_size = 32
    assert resolve_verifier_norm_width(model.config) == 32
