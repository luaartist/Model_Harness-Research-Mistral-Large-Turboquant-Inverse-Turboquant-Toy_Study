#!/usr/bin/env python3
"""GGUF Metadata Injector — DynamicFlux safety-framework pattern.

Follows the extract → validate → inject → audit pipeline from the
DynamicFlux metadata injector framework, adapted for GGUF architecture
KV pair injection.

Reads Mistral params.json, validates architecture parameters against
known bounds, converts to GGUF KV pairs that llama.cpp requires, and
maintains an injection audit log.

Architecture:
    params.json  ──▶  MistralConfigExtractor  ──▶  GGUFMetadataInjector
                         (extract + validate)          (inject + audit)
                                                           │
                                                           ▼
                                                    GGUFWriter KV pairs

The injector handles the Mistral → HF → GGUF key remapping chain:
    Mistral params.json keys  →  HF-equivalent keys  →  GGUF KV names

For Mistral Large 3 675B (MLA + MoE), the GGUF architecture is "deepseek2"
since llama.cpp reuses the DeepSeek V2 code path for MLA attention.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

GGUF_PY = Path(os.environ.get("GGUF_PY_PATH", "third_party/llama.cpp/gguf-py"))
sys.path.insert(0, str(GGUF_PY))

try:
    from gguf import GGUFWriter  # noqa: E402
except ImportError:  # pragma: no cover - optional llama.cpp dependency
    GGUFWriter = None  # type: ignore[assignment]


def require_gguf_writer() -> object:
    if GGUFWriter is None:
        raise RuntimeError(
            "gguf is not importable. Set GGUF_PY_PATH to llama.cpp/gguf-py "
            "or install the llama.cpp gguf Python package."
        )
    return GGUFWriter

logger = logging.getLogger(__name__)


# ── Injection rules engine (DynamicFlux pattern) ────────────────────────

INJECTION_RULES: dict[str, dict[str, Any]] = {
    "architecture": {
        "required_fields": [
            "dim", "n_heads", "n_kv_heads", "n_layers", "vocab_size",
            "head_dim", "norm_eps",
        ],
        "bounds_checking": {
            "dim":         {"min": 512,  "max": 32768},
            "n_heads":     {"min": 1,    "max": 256},
            "n_kv_heads":  {"min": 1,    "max": 256},
            "n_layers":    {"min": 1,    "max": 256},
            "vocab_size":  {"min": 1000, "max": 500000},
            "head_dim":    {"min": 32,   "max": 512},
            "norm_eps":    {"min": 1e-10, "max": 1e-2},
        },
    },
    "mla": {
        "required_fields": [
            "kv_lora_rank", "q_lora_rank",
            "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim",
        ],
        "bounds_checking": {
            "kv_lora_rank":     {"min": 32,  "max": 4096},
            "q_lora_rank":      {"min": 32,  "max": 4096},
            "qk_nope_head_dim": {"min": 16,  "max": 512},
            "qk_rope_head_dim": {"min": 16,  "max": 512},
            "v_head_dim":       {"min": 16,  "max": 512},
        },
    },
    "moe": {
        "required_fields": [
            "num_experts", "num_experts_per_tok",
            "expert_hidden_dim", "first_k_dense_replace",
        ],
        "bounds_checking": {
            "num_experts":          {"min": 1,   "max": 512},
            "num_experts_per_tok":  {"min": 1,   "max": 64},
            "expert_hidden_dim":    {"min": 128, "max": 65536},
            "first_k_dense_replace": {"min": 0,  "max": 256},
        },
    },
    "rope": {
        "required_fields": ["rope_theta"],
        "bounds_checking": {
            "rope_theta": {"min": 1.0, "max": 1e12},
        },
    },
}


@dataclass
class InjectionRecord:
    """Single audit record for a KV pair injection."""
    gguf_key: str
    value: Any
    source_field: str
    rule_category: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class InjectionAudit:
    """Full audit log following DynamicFlux injection_history pattern."""
    config_path: str
    extraction_time: float = 0.0
    validation_errors: list[str] = field(default_factory=list)
    validation_warnings: list[str] = field(default_factory=list)
    injections: list[InjectionRecord] = field(default_factory=list)
    total_kv_pairs: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_path": self.config_path,
            "extraction_time": self.extraction_time,
            "validation_errors": self.validation_errors,
            "validation_warnings": self.validation_warnings,
            "total_kv_pairs": self.total_kv_pairs,
            "injections": [
                {
                    "gguf_key": r.gguf_key,
                    "value": r.value if not isinstance(r.value, float) or r.value == r.value else None,
                    "source_field": r.source_field,
                    "rule_category": r.rule_category,
                }
                for r in self.injections
            ],
        }


# ── Extractor ────────────────────────────────────────────────────────────

class MistralConfigExtractor:
    """Extract and validate architecture params from Mistral params.json.

    Follows DynamicFlux pattern:
        extractor.extract()  →  validated param dict

    The extractor handles the Mistral → HF key remapping that llama.cpp's
    convert_hf_to_gguf.py performs in MistralMoeModel.__init__.
    """

    def __init__(self, config_path: Path | str):
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"config not found: {self.config_path}")
        self.raw: dict[str, Any] = {}
        self.params: dict[str, Any] = {}

    def extract(self) -> dict[str, Any]:
        """Read params.json and extract all architecture parameters."""
        self.raw = json.loads(self.config_path.read_text(encoding="utf-8"))
        cfg = self.raw

        moe = cfg.get("moe", {})

        self.params = {
            # ── Architecture core ──
            "dim":                 cfg["dim"],
            "n_heads":             cfg["n_heads"],
            "n_kv_heads":          cfg["n_kv_heads"],
            "n_layers":            cfg["n_layers"],
            "vocab_size":          cfg["vocab_size"],
            "head_dim":            cfg.get("head_dim", cfg["dim"] // cfg["n_heads"]),
            "hidden_dim":          cfg.get("hidden_dim", 0),
            "norm_eps":            cfg.get("norm_eps", 1e-6),
            "max_position_embeddings": cfg.get("max_position_embeddings", 128000),
            "tied_embeddings":     cfg.get("tied_embeddings", False),

            # ── MLA (Multi-head Latent Attention) ──
            "kv_lora_rank":        cfg.get("kv_lora_rank", 0),
            "q_lora_rank":         cfg.get("q_lora_rank", 0),
            "qk_nope_head_dim":    cfg.get("qk_nope_head_dim", 0),
            "qk_rope_head_dim":    cfg.get("qk_rope_head_dim", 0),
            "v_head_dim":          cfg.get("v_head_dim", 0),

            # ── MoE ──
            "num_experts":          moe.get("num_experts", 0),
            "num_experts_per_tok":  moe.get("num_experts_per_tok", 0),
            "expert_hidden_dim":    moe.get("expert_hidden_dim", 0),
            "first_k_dense_replace": moe.get("first_k_dense_replace", 0),
            "num_shared_experts":   moe.get("num_shared_experts", 0),
            "routed_scale":         moe.get("routed_scale", 1.0),
            "num_expert_groups":    moe.get("num_expert_groups", 1),
            "num_expert_groups_per_tok": moe.get("num_expert_groups_per_tok", 1),

            # ── RoPE ──
            "rope_theta":          cfg.get("rope_theta", 10000.0),

            # ── YaRN scaling ──
            "yarn": cfg.get("yarn", {}),
        }
        return self.params

    def validate(self) -> tuple[list[str], list[str]]:
        """Validate extracted params against injection rules.

        Returns:
            (errors, warnings) — errors are fatal, warnings are informational.
        """
        errors: list[str] = []
        warnings: list[str] = []

        for category, rules in INJECTION_RULES.items():
            # Check required fields
            for fld in rules["required_fields"]:
                val = self.params.get(fld)
                if val is None or val == 0:
                    # MLA/MoE fields may legitimately be 0 for non-MLA/non-MoE models
                    if category in ("mla", "moe"):
                        warnings.append(f"{category}.{fld} is 0/missing (non-{category} model?)")
                    else:
                        errors.append(f"required field missing or zero: {fld}")

            # Bounds checking
            for fld, bounds in rules["bounds_checking"].items():
                val = self.params.get(fld)
                if val is None or val == 0:
                    continue
                if val < bounds["min"]:
                    errors.append(f"{fld}={val} below minimum {bounds['min']}")
                if val > bounds["max"]:
                    errors.append(f"{fld}={val} above maximum {bounds['max']}")

        # Cross-field integrity checks
        if self.params["n_kv_heads"] > self.params["n_heads"]:
            errors.append(
                f"n_kv_heads ({self.params['n_kv_heads']}) > n_heads ({self.params['n_heads']})"
            )
        if self.params["num_experts_per_tok"] > self.params["num_experts"] and self.params["num_experts"] > 0:
            errors.append(
                f"num_experts_per_tok ({self.params['num_experts_per_tok']}) > "
                f"num_experts ({self.params['num_experts']})"
            )
        if self.params["first_k_dense_replace"] > self.params["n_layers"]:
            errors.append(
                f"first_k_dense_replace ({self.params['first_k_dense_replace']}) > "
                f"n_layers ({self.params['n_layers']})"
            )

        return errors, warnings


# ── Injector ─────────────────────────────────────────────────────────────

class GGUFMetadataInjector:
    """Inject validated architecture params into a GGUFWriter.

    Follows DynamicFlux pattern:
        injector.inject(writer)  →  audit log

    The injector writes ~30 GGUF KV pairs that llama.cpp needs to load
    and run inference on a Mistral Large 3 (deepseek2 arch) model.

    Key remapping chain:
        Mistral params.json  →  (MistralMoeModel remapping)  →  GGUF KV
        dim                  →  hidden_size                  →  {arch}.embedding_length
        n_heads              →  num_attention_heads           →  {arch}.attention.head_count
        ...
    """

    def __init__(self, extractor: MistralConfigExtractor, arch: str = "deepseek2"):
        self.extractor = extractor
        self.arch = arch
        self.audit = InjectionAudit(config_path=str(extractor.config_path))

    def _record(self, gguf_key: str, value: Any, source_field: str, category: str) -> None:
        """Record an injection for the audit log."""
        self.audit.injections.append(
            InjectionRecord(
                gguf_key=gguf_key,
                value=value,
                source_field=source_field,
                rule_category=category,
            )
        )

    def inject(self, writer: GGUFWriter) -> InjectionAudit:
        """Inject all architecture KV pairs into the GGUF writer.

        This is the main entry point. Call after creating the GGUFWriter.

        Args:
            writer: GGUFWriter to inject metadata into.

        Returns:
            InjectionAudit with full provenance log.

        Raises:
            RuntimeError: If validation finds fatal errors.
        """
        # ── 1. Extract ──
        t0 = time.time()
        params = self.extractor.extract()
        self.audit.extraction_time = time.time() - t0

        # ── 2. Validate ──
        errors, warnings = self.extractor.validate()
        self.audit.validation_errors = errors
        self.audit.validation_warnings = warnings

        if errors:
            msg = "Metadata validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            raise RuntimeError(msg)

        for w in warnings:
            logger.warning("metadata validation: %s", w)

        # ── 3. Inject ──
        # MLA converts to MQA (1 KV head) — same as convert_hf_to_gguf.py
        effective_n_kv_heads = 1  # DeepSeek2/Mistral MLA → MQA

        # -- General metadata --
        writer.add_name("Mistral-Large-3-675B-TurboQuant-Q4")
        self._record("general.name", "Mistral-Large-3-675B-TurboQuant-Q4",
                      "hardcoded", "general")

        writer.add_description(
            f"TurboQuant q4 decoded: {params['n_layers']}L, "
            f"{params['num_experts']}E MoE, MLA attention"
        )
        self._record("general.description", "...", "computed", "general")

        # -- Core architecture --
        writer.add_block_count(params["n_layers"])
        self._record(f"{self.arch}.block_count", params["n_layers"],
                      "n_layers", "architecture")

        writer.add_context_length(params["max_position_embeddings"])
        self._record(f"{self.arch}.context_length", params["max_position_embeddings"],
                      "max_position_embeddings", "architecture")

        writer.add_embedding_length(params["dim"])
        self._record(f"{self.arch}.embedding_length", params["dim"],
                      "dim", "architecture")

        writer.add_feed_forward_length(params["hidden_dim"])
        self._record(f"{self.arch}.feed_forward_length", params["hidden_dim"],
                      "hidden_dim", "architecture")

        writer.add_head_count(params["n_heads"])
        self._record(f"{self.arch}.attention.head_count", params["n_heads"],
                      "n_heads", "architecture")

        writer.add_head_count_kv(effective_n_kv_heads)
        self._record(f"{self.arch}.attention.head_count_kv", effective_n_kv_heads,
                      "mla→mqа=1", "architecture")

        writer.add_vocab_size(params["vocab_size"])
        self._record(f"{self.arch}.vocab_size", params["vocab_size"],
                      "vocab_size", "architecture")

        # -- Norm --
        writer.add_layer_norm_rms_eps(params["norm_eps"])
        self._record(f"{self.arch}.attention.layer_norm_rms_epsilon", params["norm_eps"],
                      "norm_eps", "architecture")

        # -- MLA (Multi-head Latent Attention) --
        if params["q_lora_rank"]:
            writer.add_q_lora_rank(params["q_lora_rank"])
            self._record(f"{self.arch}.attention.q_lora_rank", params["q_lora_rank"],
                          "q_lora_rank", "mla")

        writer.add_kv_lora_rank(params["kv_lora_rank"])
        self._record(f"{self.arch}.attention.kv_lora_rank", params["kv_lora_rank"],
                      "kv_lora_rank", "mla")

        # MLA key_length = kv_lora_rank + qk_rope_head_dim (compressed KV representation)
        key_length = params["kv_lora_rank"] + params["qk_rope_head_dim"]
        writer.add_key_length(key_length)
        self._record(f"{self.arch}.attention.key_length", key_length,
                      "kv_lora_rank + qk_rope_head_dim", "mla")

        # MLA value_length = kv_lora_rank
        writer.add_value_length(params["kv_lora_rank"])
        self._record(f"{self.arch}.attention.value_length", params["kv_lora_rank"],
                      "kv_lora_rank", "mla")

        # MLA key_length_mla = qk_nope_head_dim + qk_rope_head_dim (full decomp dim)
        key_length_mla = params["qk_nope_head_dim"] + params["qk_rope_head_dim"]
        writer.add_key_length_mla(key_length_mla)
        self._record(f"{self.arch}.attention.key_length_mla", key_length_mla,
                      "qk_nope_head_dim + qk_rope_head_dim", "mla")

        # MLA value_length_mla = v_head_dim
        writer.add_value_length_mla(params["v_head_dim"])
        self._record(f"{self.arch}.attention.value_length_mla", params["v_head_dim"],
                      "v_head_dim", "mla")

        # -- MoE --
        writer.add_expert_count(params["num_experts"])
        self._record(f"{self.arch}.expert_count", params["num_experts"],
                      "moe.num_experts", "moe")

        writer.add_expert_used_count(params["num_experts_per_tok"])
        self._record(f"{self.arch}.expert_used_count", params["num_experts_per_tok"],
                      "moe.num_experts_per_tok", "moe")

        writer.add_expert_feed_forward_length(params["expert_hidden_dim"])
        self._record(f"{self.arch}.expert_feed_forward_length", params["expert_hidden_dim"],
                      "moe.expert_hidden_dim", "moe")

        writer.add_expert_shared_count(params["num_shared_experts"])
        self._record(f"{self.arch}.expert_shared_count", params["num_shared_experts"],
                      "moe.num_shared_experts", "moe")

        writer.add_leading_dense_block_count(params["first_k_dense_replace"])
        self._record(f"{self.arch}.leading_dense_block_count", params["first_k_dense_replace"],
                      "moe.first_k_dense_replace", "moe")

        if params["routed_scale"] != 1.0:
            writer.add_expert_weights_scale(params["routed_scale"])
            self._record(f"{self.arch}.expert_weights_scale", params["routed_scale"],
                          "moe.routed_scale", "moe")

        # norm_topk_prob = True (Mistral default)
        writer.add_expert_weights_norm(True)
        self._record(f"{self.arch}.expert_weights_norm", True,
                      "default=True", "moe")

        # -- RoPE --
        writer.add_rope_freq_base(params["rope_theta"])
        self._record(f"{self.arch}.rope.freq_base", params["rope_theta"],
                      "rope_theta", "rope")

        writer.add_rope_dimension_count(params["qk_rope_head_dim"])
        self._record(f"{self.arch}.rope.dimension_count", params["qk_rope_head_dim"],
                      "qk_rope_head_dim", "rope")

        # YaRN scaling
        yarn = params.get("yarn", {})
        if yarn:
            orig_ctx = yarn.get("original_max_position_embeddings", 8192)
            writer.add_rope_scaling_orig_ctx_len(orig_ctx)
            self._record(f"{self.arch}.rope.scaling.original_context_length", orig_ctx,
                          "yarn.original_max_position_embeddings", "rope")

            # [TAG_DEEPSEEK2_YARN_LOG_MUL_FIX] — multiply by 0.1 for legacy compat
            writer.add_rope_scaling_yarn_log_mul(0.1)
            self._record(f"{self.arch}.rope.scaling.yarn_log_multiplier", 0.1,
                          "mscale_all_dim * 0.1 (legacy)", "rope")

            if "beta" in yarn:
                writer.add_rope_scaling_yarn_beta_fast(float(yarn["beta"]))
                self._record(f"{self.arch}.rope.scaling.yarn_beta_fast", yarn["beta"],
                              "yarn.beta", "rope")

            if "alpha" in yarn:
                writer.add_rope_scaling_yarn_beta_slow(float(yarn["alpha"]))
                self._record(f"{self.arch}.rope.scaling.yarn_beta_slow", yarn["alpha"],
                              "yarn.alpha", "rope")

        # -- Temperature scaling (Mistral Large) --
        if yarn:
            attn_temp_length = yarn.get("original_max_position_embeddings", 8192)
            writer.add_attn_temperature_length(attn_temp_length)
            self._record(f"{self.arch}.attention.temperature_length", attn_temp_length,
                          "yarn.original_max_position_embeddings", "architecture")

        # -- File type (F32 for decoded TurboQuant) --
        writer.add_file_type(0)  # GGML_FTYPE_ALL_F32
        self._record("general.file_type", 0, "F32 (decoded TurboQuant)", "general")

        # ── 4. Audit ──
        self.audit.total_kv_pairs = len(self.audit.injections)
        logger.info(
            "injected %d KV pairs (%d warnings)",
            self.audit.total_kv_pairs,
            len(self.audit.validation_warnings),
        )
        return self.audit


# ── Tokenizer injection ──────────────────────────────────────────────────

def inject_tekken_tokenizer(
    writer: GGUFWriter,
    model_dir: Path | str,
) -> dict[str, Any]:
    """Inject Tekken/BPE tokenizer metadata into GGUF from HuggingFace files.

    Reads tokenizer.json (BPE vocab + merges) and tokenizer_config.json
    (special token IDs), plus chat_template.jinja if present.

    Args:
        writer: GGUFWriter to inject tokenizer fields into.
        model_dir: Directory containing tokenizer.json, tokenizer_config.json,
                   and optionally chat_template.jinja.

    Returns:
        Dict summarizing what was injected.
    """
    model_dir = Path(model_dir)

    # ── Load HF tokenizer files ──
    tokenizer_path = model_dir / "tokenizer.json"
    config_path = model_dir / "tokenizer_config.json"
    template_path = model_dir / "chat_template.jinja"

    if not tokenizer_path.exists():
        raise FileNotFoundError(f"tokenizer.json not found: {tokenizer_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"tokenizer_config.json not found: {config_path}")

    tokenizer_data = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    tokenizer_config = json.loads(config_path.read_text(encoding="utf-8"))

    # ── Extract vocab (id → token) ──
    vocab_dict: dict[str, int] = tokenizer_data["model"]["vocab"]
    merges_raw: list[list[str]] = tokenizer_data["model"]["merges"]
    added_tokens: list[dict] = tokenizer_data.get("added_tokens", [])

    # Build id→token mapping
    id_to_token: dict[int, str] = {v: k for k, v in vocab_dict.items()}

    # Added tokens may override base vocab entries (special tokens occupy IDs 0-999)
    special_ids: set[int] = set()
    for at in added_tokens:
        id_to_token[at["id"]] = at["content"]
        if at.get("special", False):
            special_ids.add(at["id"])

    # Build ordered token list and type list
    max_id = max(id_to_token.keys())
    tokens: list[str] = []
    token_types: list[int] = []

    for i in range(max_id + 1):
        if i in id_to_token:
            tokens.append(id_to_token[i])
            if i in special_ids:
                token_types.append(3)  # CONTROL
            else:
                token_types.append(1)  # NORMAL
        else:
            tokens.append(f"[PAD{i}]")
            token_types.append(5)  # UNUSED

    # Build merges list (space-joined pairs)
    merges: list[str] = [f"{pair[0]} {pair[1]}" for pair in merges_raw]

    # ── Special token IDs ──
    bos_id = vocab_dict.get("<s>", 1)
    eos_id = vocab_dict.get("</s>", 2)
    unk_id = vocab_dict.get("<unk>", 0)
    pad_id = vocab_dict.get("<pad>", None)

    # ── Chat template ──
    chat_template = ""
    if template_path.exists():
        chat_template = template_path.read_text(encoding="utf-8")
        logger.info("loaded chat_template.jinja (%d chars)", len(chat_template))
    else:
        chat_template = tokenizer_config.get("chat_template", "")

    # ── Write to GGUF ──
    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre("tekken")
    writer.add_token_list(tokens)
    writer.add_token_types(token_types)
    writer.add_token_merges(merges)
    writer.add_bos_token_id(bos_id)
    writer.add_eos_token_id(eos_id)
    writer.add_unk_token_id(unk_id)
    if pad_id is not None:
        writer.add_pad_token_id(pad_id)
    writer.add_add_bos_token(tokenizer_config.get("add_bos_token", True))
    writer.add_add_eos_token(tokenizer_config.get("add_eos_token", False))
    if chat_template:
        writer.add_chat_template(chat_template)

    summary = {
        "vocab_size": len(tokens),
        "merges": len(merges),
        "special_tokens": len(special_ids),
        "bos_id": bos_id,
        "eos_id": eos_id,
        "unk_id": unk_id,
        "pad_id": pad_id,
        "has_chat_template": bool(chat_template),
    }
    logger.info(
        "tokenizer injected: %d tokens, %d merges, %d special, chat_template=%s",
        len(tokens), len(merges), len(special_ids), bool(chat_template),
    )
    return summary


# ── Convenience function for merger integration ──────────────────────────

def inject_mistral_metadata(
    writer: GGUFWriter,
    config_path: Path | str,
    arch: str = "deepseek2",
) -> InjectionAudit:
    """One-call entry point: extract, validate, inject, return audit.

    Args:
        writer: GGUFWriter that has already been created with the correct arch.
        config_path: Path to Mistral params.json.
        arch: GGUF architecture name (default: deepseek2 for MLA models).

    Returns:
        InjectionAudit with full provenance log.
    """
    extractor = MistralConfigExtractor(config_path)
    injector = GGUFMetadataInjector(extractor, arch=arch)
    return injector.inject(writer)


# ── CLI for standalone testing ───────────────────────────────────────────

def main() -> None:
    """CLI: validate a params.json and print what would be injected."""
    import argparse

    parser = argparse.ArgumentParser(description="GGUF Metadata Injector (dry-run)")
    parser.add_argument(
        "config",
        type=Path,
        help="Path to Mistral params.json",
    )
    parser.add_argument(
        "--arch",
        default="deepseek2",
        help="GGUF architecture (default: deepseek2)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    extractor = MistralConfigExtractor(args.config)
    params = extractor.extract()

    print(f"config: {args.config}")
    print(f"arch:   {args.arch}")
    print(f"\nextracted {len(params)} parameters:")
    for k, v in sorted(params.items()):
        if isinstance(v, dict):
            print(f"  {k}: {{...}}")
        else:
            print(f"  {k}: {v}")

    errors, warnings = extractor.validate()
    if warnings:
        print(f"\nwarnings ({len(warnings)}):")
        for w in warnings:
            print(f"  ⚠ {w}")
    if errors:
        print(f"\nERRORS ({len(errors)}):")
        for e in errors:
            print(f"  ✗ {e}")
        raise SystemExit(1)

    print(f"\n✓ validation passed ({len(warnings)} warnings)")

    # Dry-run injection — write to /dev/null equivalent
    tmp_path = Path("/tmp/_gguf_injector_dryrun.gguf")
    writer_cls = require_gguf_writer()
    writer = writer_cls(tmp_path, arch=args.arch)
    injector = GGUFMetadataInjector(extractor, arch=args.arch)
    audit = injector.inject(writer)
    writer.close()
    tmp_path.unlink(missing_ok=True)

    print(f"\ninjection audit ({audit.total_kv_pairs} KV pairs):")
    for rec in audit.injections:
        val_str = repr(rec.value) if not isinstance(rec.value, str) else rec.value
        print(f"  {rec.gguf_key:50s} = {val_str:>20s}  ← {rec.source_field}")

    # Write audit JSON
    audit_path = args.config.parent / "gguf_injection_audit.json"
    audit_path.write_text(json.dumps(audit.to_dict(), indent=2), encoding="utf-8")
    print(f"\naudit log: {audit_path}")


if __name__ == "__main__":
    main()
