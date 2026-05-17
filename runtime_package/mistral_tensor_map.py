#!/usr/bin/env python3
"""Mistral Large 3 → GGUF tensor name mapping.

Maps Mistral safetensor name patterns to llama.cpp GGUF names.
Based on llama.cpp's gguf-py/gguf/tensor_mapping.py (DeepSeek2/MLA arch).

Mistral Large 3 (675B) has:
  - 61 layers (3 dense + 58 MoE, first_k_dense_replace=3)
  - Layers 0-2: dense feed_forward (w1/w2/w3)
  - Layers 3-60: MoE with 128 experts + gate router + 1 shared expert
  - MLA attention: wq_a/wq_b (latent query), wkv_a_with_mqa/wkv_b (latent KV)
  - Mixed dtypes: bfloat16 (norms, wq_a, wkv_a, gate, embeddings) + float8_e4m3fn (weights)
  - weight_scale tensors for float8 weights (consumed during dequant, not output)
  - Non-layer tensors: tok_embeddings, output, norm

Usage:
    from mistral_tensor_map import mistral_to_gguf, list_all_mistral_tensors

    gguf_name = mistral_to_gguf("layers.5.attention.wq_a.weight")
    # -> "blk.5.attn_q_a.weight"
    gguf_name = mistral_to_gguf("tok_embeddings.weight")
    # -> "token_embd.weight"
"""

from __future__ import annotations

import re
from typing import Optional


# ─────────────────────────────────────────────────────────────────────
# Non-layer tensors: direct name → name mapping
# ─────────────────────────────────────────────────────────────────────
_SPECIAL_MAP: dict[str, str] = {
    "tok_embeddings.weight": "token_embd.weight",
    "output.weight":         "output.weight",
    "norm.weight":           "output_norm.weight",
}


# ─────────────────────────────────────────────────────────────────────
# Layer-level mapping: Mistral safetensor suffix → GGUF suffix
# Derived from llama.cpp/gguf-py/gguf/tensor_mapping.py (DeepSeek2 arch)
# ─────────────────────────────────────────────────────────────────────
_MISTRAL_TO_GGUF_SUFFIX: dict[str, str] = {
    # Attention (MLA architecture)
    "attention.wq_a.weight":              "attn_q_a.weight",
    "attention.wq_b.weight":              "attn_q_b.weight",
    "attention.wq_b.weight_scale":        "attn_q_b.weight_scale",
    "attention.q_a_norm.weight":          "attn_q_a_norm.weight",
    "attention.wkv_a_with_mqa.weight":    "attn_kv_a_mqa.weight",
    "attention.wkv_b.weight":             "attn_kv_b.weight",
    "attention.wkv_b.weight_scale":       "attn_kv_b.weight_scale",
    "attention.kv_a_norm.weight":         "attn_kv_a_norm.weight",
    "attention.wo.weight":                "attn_output.weight",
    "attention.wo.weight_scale":          "attn_output.weight_scale",
    "attention_norm.weight":              "attn_norm.weight",

    # Dense feed-forward (layers 0-2)
    "feed_forward.w1.weight":             "ffn_gate.weight",
    "feed_forward.w1.weight_scale":       "ffn_gate.weight_scale",
    "feed_forward.w2.weight":             "ffn_down.weight",
    "feed_forward.w2.weight_scale":       "ffn_down.weight_scale",
    "feed_forward.w3.weight":             "ffn_up.weight",
    "feed_forward.w3.weight_scale":       "ffn_up.weight_scale",
    "ffn_norm.weight":                    "ffn_norm.weight",

    # MoE router gate (layers 3-60)
    "gate.weight":                        "ffn_gate_inp.weight",

    # Shared experts (layers 3-60) — single shared expert per layer
    "shared_experts.w1.weight":           "ffn_gate_shexp.weight",
    "shared_experts.w1.weight_scale":     "ffn_gate_shexp.weight_scale",
    "shared_experts.w2.weight":           "ffn_down_shexp.weight",
    "shared_experts.w2.weight_scale":     "ffn_down_shexp.weight_scale",
    "shared_experts.w3.weight":           "ffn_up_shexp.weight",
    "shared_experts.w3.weight_scale":     "ffn_up_shexp.weight_scale",

    # MoE experts (layers 3-60) — individual: experts.{E}.w{1,2,3}
    "w1.weight":                          "ffn_gate_exp.weight",
    "w1.weight_scale":                    "ffn_gate_exp.weight_scale",
    "w2.weight":                          "ffn_down_exp.weight",
    "w2.weight_scale":                    "ffn_down_exp.weight_scale",
    "w3.weight":                          "ffn_up_exp.weight",
    "w3.weight_scale":                    "ffn_up_exp.weight_scale",
}

# Regex: layers.{layer}.experts.{expert}.w{n}.weight[_scale]
_EXPERT_RE = re.compile(
    r"^layers\.(\d+)\.experts\.(\d+)\.(w[123]\.weight(?:_scale)?)$"
)

# Regex: layers.{layer}.{suffix}
_LAYER_RE = re.compile(r"^layers\.(\d+)\.(.+)$")


def mistral_to_gguf(mistral_name: str) -> Optional[str]:
    """Convert a Mistral safetensor name to GGUF convention.

    Returns None if the name doesn't match any known pattern.

    Examples:
        tok_embeddings.weight                -> token_embd.weight
        output.weight                        -> output.weight
        norm.weight                          -> output_norm.weight
        layers.5.attention.wq_a.weight       -> blk.5.attn_q_a.weight
        layers.10.experts.42.w1.weight       -> blk.10.ffn_gate_exp.42.weight
        layers.0.feed_forward.w1.weight      -> blk.0.ffn_gate.weight
        layers.10.gate.weight                -> blk.10.ffn_gate_inp.weight
        layers.10.shared_experts.w1.weight   -> blk.10.ffn_gate_shexp.weight
        layers.100.attention_norm.weight     -> blk.100.attn_norm.weight
    """
    # Check non-layer (special) tensors first
    if mistral_name in _SPECIAL_MAP:
        return _SPECIAL_MAP[mistral_name]

    # Try expert pattern (most specific)
    m = _EXPERT_RE.match(mistral_name)
    if m:
        layer, expert, suffix = m.group(1), m.group(2), m.group(3)
        gguf_suffix = _MISTRAL_TO_GGUF_SUFFIX.get(suffix)
        if gguf_suffix is None:
            return None
        # Insert expert index: blk.N.ffn_gate_exp.E.weight
        base, ext = gguf_suffix.rsplit(".", 1)
        return f"blk.{layer}.{base}.{expert}.{ext}"

    # Try standard layer pattern
    m = _LAYER_RE.match(mistral_name)
    if m:
        layer, suffix = m.group(1), m.group(2)
        gguf_suffix = _MISTRAL_TO_GGUF_SUFFIX.get(suffix)
        if gguf_suffix is None:
            return None
        return f"blk.{layer}.{gguf_suffix}"

    return None


def gguf_to_mistral(gguf_name: str) -> Optional[str]:
    """Reverse map: GGUF → Mistral safetensor name.

    Returns None if no reverse mapping is found.
    """
    # Build reverse maps lazily
    if not hasattr(gguf_to_mistral, "_reverse"):
        gguf_to_mistral._reverse = {v: k for k, v in _MISTRAL_TO_GGUF_SUFFIX.items()}
        gguf_to_mistral._special_reverse = {v: k for k, v in _SPECIAL_MAP.items()}

    # Check non-layer (special) tensors
    if gguf_name in gguf_to_mistral._special_reverse:
        return gguf_to_mistral._special_reverse[gguf_name]

    # Expert pattern: blk.N.ffn_gate_exp.E.weight
    expert_re = re.compile(r"^blk\.(\d+)\.(ffn_\w+_exp)\.(\d+)\.(weight(?:_scale)?)$")
    m = expert_re.match(gguf_name)
    if m:
        layer, base, expert, ext = m.groups()
        gguf_suffix = f"{base}.{ext}"
        mistral_suffix = gguf_to_mistral._reverse.get(gguf_suffix)
        if mistral_suffix:
            return f"layers.{layer}.experts.{expert}.{mistral_suffix}"
        return None

    # Standard: blk.N.suffix
    layer_re = re.compile(r"^blk\.(\d+)\.(.+)$")
    m = layer_re.match(gguf_name)
    if m:
        layer, suffix = m.groups()
        mistral_suffix = gguf_to_mistral._reverse.get(suffix)
        if mistral_suffix:
            return f"layers.{layer}.{mistral_suffix}"
    return None


def list_all_mistral_tensors(n_layers: int = 61, n_experts: int = 128,
                              dense_layers: int = 3) -> list[str]:
    """Generate all expected Mistral tensor names for the full model.

    Args:
        n_layers: Total number of layers (61 for Mistral Large 3 675B)
        n_experts: Number of experts per MoE layer (128)
        dense_layers: Number of initial dense FFN layers (3, first_k_dense_replace)

    Returns:
        Sorted list of all tensor names.
    """
    names: list[str] = []

    # Non-layer tensors
    names.append("tok_embeddings.weight")
    names.append("output.weight")
    names.append("norm.weight")

    for layer in range(n_layers):
        # Attention tensors (all layers)
        for suffix in [
            "attention.wq_a.weight",
            "attention.wq_b.weight",
            "attention.wq_b.weight_scale",
            "attention.q_a_norm.weight",
            "attention.wkv_a_with_mqa.weight",
            "attention.wkv_b.weight",
            "attention.wkv_b.weight_scale",
            "attention.kv_a_norm.weight",
            "attention.wo.weight",
            "attention.wo.weight_scale",
            "attention_norm.weight",
        ]:
            names.append(f"layers.{layer}.{suffix}")

        if layer < dense_layers:
            # Dense feed-forward layers
            for suffix in [
                "feed_forward.w1.weight",
                "feed_forward.w1.weight_scale",
                "feed_forward.w2.weight",
                "feed_forward.w2.weight_scale",
                "feed_forward.w3.weight",
                "feed_forward.w3.weight_scale",
            ]:
                names.append(f"layers.{layer}.{suffix}")
        else:
            # MoE layers: router + shared experts + routed experts
            names.append(f"layers.{layer}.gate.weight")
            for suffix in [
                "shared_experts.w1.weight",
                "shared_experts.w1.weight_scale",
                "shared_experts.w2.weight",
                "shared_experts.w2.weight_scale",
                "shared_experts.w3.weight",
                "shared_experts.w3.weight_scale",
            ]:
                names.append(f"layers.{layer}.{suffix}")
            for expert in range(n_experts):
                for wn in ["w1", "w2", "w3"]:
                    names.append(f"layers.{layer}.experts.{expert}.{wn}.weight")
                    names.append(f"layers.{layer}.experts.{expert}.{wn}.weight_scale")

        # FFN norm (all layers)
        names.append(f"layers.{layer}.ffn_norm.weight")

    return sorted(names)


def _self_test() -> None:
    """Verify all patterns map correctly."""
    test_cases = [
        # Non-layer tensors
        ("tok_embeddings.weight", "token_embd.weight"),
        ("output.weight", "output.weight"),
        ("norm.weight", "output_norm.weight"),
        # Attention
        ("layers.0.attention.wq_a.weight", "blk.0.attn_q_a.weight"),
        ("layers.5.attention.wq_b.weight", "blk.5.attn_q_b.weight"),
        ("layers.0.attention.wkv_a_with_mqa.weight", "blk.0.attn_kv_a_mqa.weight"),
        ("layers.0.attention.wkv_b.weight", "blk.0.attn_kv_b.weight"),
        ("layers.0.attention.wo.weight", "blk.0.attn_output.weight"),
        ("layers.0.attention_norm.weight", "blk.0.attn_norm.weight"),
        ("layers.0.attention.q_a_norm.weight", "blk.0.attn_q_a_norm.weight"),
        ("layers.0.attention.kv_a_norm.weight", "blk.0.attn_kv_a_norm.weight"),
        # Dense FFN
        ("layers.0.feed_forward.w1.weight", "blk.0.ffn_gate.weight"),
        ("layers.0.feed_forward.w2.weight", "blk.0.ffn_down.weight"),
        ("layers.0.feed_forward.w3.weight", "blk.0.ffn_up.weight"),
        ("layers.0.ffn_norm.weight", "blk.0.ffn_norm.weight"),
        # MoE router
        ("layers.10.gate.weight", "blk.10.ffn_gate_inp.weight"),
        # Shared experts
        ("layers.10.shared_experts.w1.weight", "blk.10.ffn_gate_shexp.weight"),
        ("layers.10.shared_experts.w2.weight", "blk.10.ffn_down_shexp.weight"),
        ("layers.10.shared_experts.w3.weight", "blk.10.ffn_up_shexp.weight"),
        ("layers.10.shared_experts.w1.weight_scale", "blk.10.ffn_gate_shexp.weight_scale"),
        # Routed experts
        ("layers.10.experts.42.w1.weight", "blk.10.ffn_gate_exp.42.weight"),
        ("layers.10.experts.0.w2.weight", "blk.10.ffn_down_exp.0.weight"),
        ("layers.10.experts.127.w3.weight", "blk.10.ffn_up_exp.127.weight"),
        ("layers.10.experts.42.w1.weight_scale", "blk.10.ffn_gate_exp.42.weight_scale"),
        # Weight scales
        ("layers.0.attention.wq_b.weight_scale", "blk.0.attn_q_b.weight_scale"),
        ("layers.0.attention.wo.weight_scale", "blk.0.attn_output.weight_scale"),
        ("layers.0.feed_forward.w1.weight_scale", "blk.0.ffn_gate.weight_scale"),
    ]

    passed = 0
    for mistral, expected_gguf in test_cases:
        result = mistral_to_gguf(mistral)
        if result != expected_gguf:
            print(f"  FAIL: {mistral}")
            print(f"    expected: {expected_gguf}")
            print(f"    got:      {result}")
        else:
            passed += 1

    # Test reverse mapping
    for mistral, gguf_name in test_cases:
        reverse = gguf_to_mistral(gguf_name)
        if reverse != mistral:
            print(f"  REVERSE FAIL: {gguf_name}")
            print(f"    expected: {mistral}")
            print(f"    got:      {reverse}")
        else:
            passed += 1

    total = len(test_cases) * 2
    print(f"mistral_tensor_map: {passed}/{total} tests passed")

    # Verify total tensor count
    all_tensors = list_all_mistral_tensors()
    all_mapped = [(t, mistral_to_gguf(t)) for t in all_tensors]
    unmapped = [t for t, g in all_mapped if g is None]
    if unmapped:
        print(f"  WARNING: {len(unmapped)} unmapped tensors")
        for t in unmapped[:5]:
            print(f"    {t}")
    else:
        print(f"  all {len(all_tensors)} tensors map successfully")

    if passed < total:
        raise SystemExit(1)


if __name__ == "__main__":
    _self_test()
