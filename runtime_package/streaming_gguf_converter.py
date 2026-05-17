#!/usr/bin/env python3
"""Phase 4: Streaming safetensor → GGUF converter for Mistral Large 3 675B.

Converts 272 safetensor shards (681.5 GB, fp8+bf16) into a single GGUF file
with F16 tensors. Uses a 2-pass streaming approach:

  Pass 1 (Planning):  Read index.json → compute all output tensor metadata
                       → register via add_tensor_info() → write GGUF header
  Pass 2 (Streaming):  For each tensor (in registration order):
                       → load from shard → dequant fp8 → process → write

Key transformations:
  - fp8_e4m3fn × weight_scale → f16 (block-wise dequant, block_size=[128,128])
  - wkv_b split into k_b (transposed) + v_b (MLA absorption)
  - 128 expert tensors stacked into 3D [n_experts, rows, cols]
  - Vision tensors skipped (text-only inference)
  - weight_scale tensors consumed (not written to GGUF)

Memory budget: ~8 GB peak (one stacked expert tensor in f16).
Disk: writes ~1.35 TB F16 GGUF, suitable for llama-quantize → Q4_K_M.

Requires: torch, numpy, safetensors, gguf (from llama.cpp gguf-py)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import struct
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from safetensors import safe_open

# gguf-py must be on path
GGUF_PY_PATH = os.environ.get(
    "GGUF_PY_PATH",
    "third_party/llama.cpp/gguf-py",
)
if GGUF_PY_PATH not in sys.path:
    sys.path.insert(0, GGUF_PY_PATH)

try:
    import gguf  # noqa: E402,F401
    from gguf import GGUFWriter, GGMLQuantizationType  # noqa: E402
except ImportError:  # pragma: no cover - optional llama.cpp dependency
    GGUFWriter = None  # type: ignore[assignment]
    GGMLQuantizationType = None  # type: ignore[assignment]


def require_gguf_writer() -> object:
    if GGUFWriter is None:
        raise RuntimeError(
            "gguf is not importable. Set GGUF_PY_PATH to llama.cpp/gguf-py "
            "or install the llama.cpp gguf Python package."
        )
    return GGUFWriter

# Local imports
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from mistral_tensor_map import mistral_to_gguf  # noqa: E402
from gguf_metadata_injector import inject_mistral_metadata, inject_tekken_tokenizer  # noqa: E402

logger = logging.getLogger("streaming_gguf_converter")


# ─────────────────────────────────────────────────────────────────────
# Architecture constants (Mistral Large 3 675B)
# ─────────────────────────────────────────────────────────────────────
N_LAYERS = 61
FIRST_K_DENSE = 3
N_EXPERTS = 128
N_KV_HEADS = 128
QK_NOPE_HEAD_DIM = 128
V_HEAD_DIM = 128
KV_LORA_RANK = 512
FP8_BLOCK_SIZE = [128, 128]


# ─────────────────────────────────────────────────────────────────────
# Output tensor entry — one per tensor in the final GGUF
# ─────────────────────────────────────────────────────────────────────
@dataclass
class OutputTensor:
    """Describes one tensor to be written to the GGUF output."""
    gguf_name: str
    shape: tuple[int, ...]
    dtype: np.dtype
    nbytes: int
    # Source info for loading
    kind: str  # "direct", "dequant", "kv_split_k", "kv_split_v", "expert_stack"
    sources: list[str] = field(default_factory=list)  # Mistral tensor name(s)
    layer: int = -1


# ─────────────────────────────────────────────────────────────────────
# FP8 dequantization
# ─────────────────────────────────────────────────────────────────────
def dequant_fp8(
    weight: torch.Tensor,
    scale: torch.Tensor,
    block_size: list[int] | None = None,
) -> torch.Tensor:
    """Dequantize fp8 weight using scale, matching llama.cpp's dequant_simple.

    Args:
        weight: fp8_e4m3fn tensor
        scale: bf16/f32 scale tensor (per-block)
        block_size: [row_block, col_block] for block-wise scaling

    Returns:
        float32 tensor (caller converts to f16)
    """
    scale = scale.float()
    if block_size is not None:
        for i, size in enumerate(block_size):
            scale = scale.repeat_interleave(size, dim=i)
        # Trim to match weight shape (in case of padding)
        scale = scale[tuple(slice(0, s) for s in weight.shape)]
    while scale.ndim < weight.ndim:
        scale = scale.unsqueeze(-1)
    return weight.float() * scale


# ─────────────────────────────────────────────────────────────────────
# Shard cache — lazy open/close of safetensor files
# ─────────────────────────────────────────────────────────────────────
class ShardCache:
    """Manages safe_open handles with LRU eviction."""

    def __init__(self, model_dir: Path, max_open: int = 8):
        self._model_dir = model_dir
        self._max_open = max_open
        self._handles: dict[str, object] = {}
        self._access_order: list[str] = []

    def get_tensor(self, shard_name: str, tensor_name: str) -> torch.Tensor:
        """Load a single tensor from a shard file."""
        if shard_name not in self._handles:
            if len(self._handles) >= self._max_open:
                # Evict LRU
                evict = self._access_order.pop(0)
                del self._handles[evict]
            path = self._model_dir / shard_name
            self._handles[shard_name] = safe_open(str(path), framework="pt")
        # Update access order
        if shard_name in self._access_order:
            self._access_order.remove(shard_name)
        self._access_order.append(shard_name)
        return self._handles[shard_name].get_tensor(tensor_name)

    def close_all(self):
        self._handles.clear()
        self._access_order.clear()


# ─────────────────────────────────────────────────────────────────────
# Streaming GGUF Converter
# ─────────────────────────────────────────────────────────────────────
class StreamingGGUFConverter:
    """Converts Mistral Large 3 safetensors → single GGUF file (F16).

    Uses GGUFWriter's add_tensor_info() + write_tensor_data() for streaming.
    Peak memory: ~8 GB (one stacked expert tensor).
    """

    def __init__(
        self,
        model_dir: Path,
        output_path: Path,
        config_path: Path,
        arch: str = "deepseek2",
        n_layers: int = N_LAYERS,
        first_k_dense: int = FIRST_K_DENSE,
        n_experts: int = N_EXPERTS,
        max_shard_handles: int = 8,
    ):
        self.model_dir = model_dir
        self.output_path = output_path
        self.config_path = config_path
        self.arch = arch
        self.n_layers = n_layers
        self.first_k_dense = first_k_dense
        self.n_experts = n_experts
        self.shard_cache = ShardCache(model_dir, max_open=max_shard_handles)

        # Load index
        index_path = model_dir / "consolidated.safetensors.index.json"
        with open(index_path) as f:
            idx = json.load(f)
        self.weight_map: dict[str, str] = idx["weight_map"]

        # Pre-read tensor shapes/dtypes from shard headers
        self._tensor_meta: dict[str, tuple[list[int], str]] = {}
        self._read_all_metadata()

        # Ordered output plan
        self.plan: list[OutputTensor] = []

    def _read_all_metadata(self):
        """Read shape/dtype metadata from safetensor headers (no data loaded)."""
        logger.info("Reading tensor metadata from shard headers...")
        seen_shards: set[str] = set()
        for tensor_name, shard_name in self.weight_map.items():
            if shard_name in seen_shards:
                continue
            seen_shards.add(shard_name)
            path = self.model_dir / shard_name
            with safe_open(str(path), framework="pt") as f:
                for key in f.keys():
                    sl = f.get_slice(key)
                    shape = sl.get_shape()
                    dtype = sl.get_dtype()
                    self._tensor_meta[key] = (shape, dtype)
            if len(seen_shards) % 50 == 0:
                logger.info(f"  ...read {len(seen_shards)}/272 shard headers")
        logger.info(f"  Total tensor metadata entries: {len(self._tensor_meta)}")

    def _is_fp8(self, tensor_name: str) -> bool:
        """Check if a tensor is fp8 quantized."""
        if tensor_name not in self._tensor_meta:
            return False
        _, dtype = self._tensor_meta[tensor_name]
        return "float8" in dtype

    def _get_shape(self, tensor_name: str) -> list[int]:
        if tensor_name not in self._tensor_meta:
            raise KeyError(f"Tensor {tensor_name!r} not found in metadata")
        return self._tensor_meta[tensor_name][0]

    def _get_scale_name(self, weight_name: str) -> Optional[str]:
        """Get the weight_scale tensor name for an fp8 weight, if it exists."""
        scale_name = weight_name.replace(".weight", ".weight_scale")
        if scale_name in self.weight_map:
            return scale_name
        return None

    # ─── Plan building ───────────────────────────────────────────────

    def _build_plan(self):
        """Build the ordered output tensor plan.

        This determines both the registration order and write order.
        """
        self.plan.clear()
        f16_dtype = np.dtype(np.float16)

        def _f16_nbytes(shape: tuple[int, ...]) -> int:
            return int(np.prod(shape)) * 2

        # 1. token_embd.weight (bf16 → f16)
        shape = tuple(self._get_shape("tok_embeddings.weight"))
        self.plan.append(OutputTensor(
            gguf_name="token_embd.weight", shape=shape, dtype=f16_dtype,
            nbytes=_f16_nbytes(shape), kind="direct",
            sources=["tok_embeddings.weight"],
        ))

        # 2. Per-layer tensors
        for layer in range(self.n_layers):
            self._plan_layer(layer, f16_dtype)

        # 3. output_norm.weight
        shape = tuple(self._get_shape("norm.weight"))
        self.plan.append(OutputTensor(
            gguf_name="output_norm.weight", shape=shape, dtype=f16_dtype,
            nbytes=_f16_nbytes(shape), kind="direct",
            sources=["norm.weight"],
        ))

        # 4. output.weight
        shape = tuple(self._get_shape("output.weight"))
        self.plan.append(OutputTensor(
            gguf_name="output.weight", shape=shape, dtype=f16_dtype,
            nbytes=_f16_nbytes(shape), kind="direct",
            sources=["output.weight"],
        ))

        logger.info(f"Plan: {len(self.plan)} output tensors")

    def _plan_layer(self, layer: int, f16_dtype: np.dtype):
        """Add all output tensors for one layer to the plan."""

        def _f16_nbytes(shape: tuple[int, ...]) -> int:
            return int(np.prod(shape)) * 2

        def _add(gguf_name: str, shape: tuple[int, ...], kind: str,
                 sources: list[str]):
            self.plan.append(OutputTensor(
                gguf_name=gguf_name, shape=shape, dtype=f16_dtype,
                nbytes=_f16_nbytes(shape), kind=kind, sources=sources,
                layer=layer,
            ))

        L = layer

        # Norms (1D, bf16)
        _add(f"blk.{L}.attn_norm.weight",
             tuple(self._get_shape(f"layers.{L}.attention_norm.weight")),
             "direct", [f"layers.{L}.attention_norm.weight"])

        _add(f"blk.{L}.attn_q_a_norm.weight",
             tuple(self._get_shape(f"layers.{L}.attention.q_a_norm.weight")),
             "direct", [f"layers.{L}.attention.q_a_norm.weight"])

        _add(f"blk.{L}.attn_kv_a_norm.weight",
             tuple(self._get_shape(f"layers.{L}.attention.kv_a_norm.weight")),
             "direct", [f"layers.{L}.attention.kv_a_norm.weight"])

        # wq_a (bf16, not fp8)
        _add(f"blk.{L}.attn_q_a.weight",
             tuple(self._get_shape(f"layers.{L}.attention.wq_a.weight")),
             "direct", [f"layers.{L}.attention.wq_a.weight"])

        # wq_b (fp8 → dequant → f16)
        _add(f"blk.{L}.attn_q_b.weight",
             tuple(self._get_shape(f"layers.{L}.attention.wq_b.weight")),
             "dequant", [f"layers.{L}.attention.wq_b.weight"])

        # wkv_a_with_mqa (bf16, not fp8)
        _add(f"blk.{L}.attn_kv_a_mqa.weight",
             tuple(self._get_shape(f"layers.{L}.attention.wkv_a_with_mqa.weight")),
             "direct", [f"layers.{L}.attention.wkv_a_with_mqa.weight"])

        # wkv_b → split into k_b (transposed) + v_b
        wkv_b_shape = self._get_shape(f"layers.{L}.attention.wkv_b.weight")
        # After split: k_b [n_kv, qk_nope, lora_rank] → transpose → [n_kv, lora_rank, qk_nope]
        k_b_shape = (N_KV_HEADS, wkv_b_shape[1], QK_NOPE_HEAD_DIM)
        v_b_shape = (N_KV_HEADS, V_HEAD_DIM, wkv_b_shape[1])
        _add(f"blk.{L}.attn_k_b.weight", k_b_shape, "kv_split_k",
             [f"layers.{L}.attention.wkv_b.weight"])
        _add(f"blk.{L}.attn_v_b.weight", v_b_shape, "kv_split_v",
             [f"layers.{L}.attention.wkv_b.weight"])

        # wo (fp8 → dequant → f16)
        _add(f"blk.{L}.attn_output.weight",
             tuple(self._get_shape(f"layers.{L}.attention.wo.weight")),
             "dequant", [f"layers.{L}.attention.wo.weight"])

        # ffn_norm
        _add(f"blk.{L}.ffn_norm.weight",
             tuple(self._get_shape(f"layers.{L}.ffn_norm.weight")),
             "direct", [f"layers.{L}.ffn_norm.weight"])

        if layer < self.first_k_dense:
            # Dense FFN
            for wn, gguf_prefix in [("w1", "ffn_gate"), ("w2", "ffn_down"), ("w3", "ffn_up")]:
                src = f"layers.{L}.feed_forward.{wn}.weight"
                _add(f"blk.{L}.{gguf_prefix}.weight",
                     tuple(self._get_shape(src)), "dequant", [src])
        else:
            # MoE layer
            # Router gate (bf16)
            _add(f"blk.{L}.ffn_gate_inp.weight",
                 tuple(self._get_shape(f"layers.{L}.gate.weight")),
                 "direct", [f"layers.{L}.gate.weight"])

            # Stacked experts: 128 individual → 1 stacked per weight type
            for wn, gguf_prefix in [("w1", "ffn_gate_exps"), ("w2", "ffn_down_exps"), ("w3", "ffn_up_exps")]:
                # Individual expert shape
                exp_shape = self._get_shape(f"layers.{L}.experts.0.{wn}.weight")
                stacked_shape = (self.n_experts, exp_shape[0], exp_shape[1])
                sources = [f"layers.{L}.experts.{e}.{wn}.weight"
                           for e in range(self.n_experts)]
                _add(f"blk.{L}.{gguf_prefix}.weight", stacked_shape,
                     "expert_stack", sources)

            # Shared experts (fp8 → dequant → f16)
            for wn, gguf_prefix in [("w1", "ffn_gate_shexp"), ("w2", "ffn_down_shexp"), ("w3", "ffn_up_shexp")]:
                src = f"layers.{L}.shared_experts.{wn}.weight"
                _add(f"blk.{L}.{gguf_prefix}.weight",
                     tuple(self._get_shape(src)), "dequant", [src])

    # ─── Tensor loading & processing ──────────────────────────────────

    def _load_tensor(self, name: str) -> torch.Tensor:
        """Load a single tensor from its shard."""
        shard = self.weight_map[name]
        return self.shard_cache.get_tensor(shard, name)

    def _dequant_tensor(self, weight_name: str) -> torch.Tensor:
        """Load fp8 weight + scale, dequantize to float32."""
        weight = self._load_tensor(weight_name)
        scale_name = self._get_scale_name(weight_name)
        if scale_name is None:
            raise ValueError(f"No weight_scale for fp8 tensor {weight_name!r}")
        scale = self._load_tensor(scale_name)
        return dequant_fp8(weight, scale, block_size=FP8_BLOCK_SIZE)

    def _process_tensor(self, entry: OutputTensor) -> np.ndarray:
        """Load, process, and return a numpy f16 array for one output tensor."""
        if entry.kind == "direct":
            # bf16 → f16
            t = self._load_tensor(entry.sources[0])
            return t.half().numpy()

        elif entry.kind == "dequant":
            # fp8 × scale → f32 → f16
            t = self._dequant_tensor(entry.sources[0])
            return t.half().numpy()

        elif entry.kind == "kv_split_k":
            # wkv_b → dequant → reshape → split → transpose k_b
            wkv_b_name = entry.sources[0]
            wkv_b = self._dequant_tensor(wkv_b_name)
            # Reshape to [n_kv_heads, qk_nope + v_head, kv_lora_rank]
            kv_b = wkv_b.view(N_KV_HEADS, QK_NOPE_HEAD_DIM + V_HEAD_DIM, -1)
            k_b, _ = torch.split(kv_b, [QK_NOPE_HEAD_DIM, V_HEAD_DIM], dim=1)
            k_b = k_b.transpose(1, 2)  # [n_kv, lora_rank, qk_nope]
            return k_b.half().contiguous().numpy()

        elif entry.kind == "kv_split_v":
            # wkv_b → dequant → reshape → split → v_b
            wkv_b_name = entry.sources[0]
            wkv_b = self._dequant_tensor(wkv_b_name)
            kv_b = wkv_b.view(N_KV_HEADS, QK_NOPE_HEAD_DIM + V_HEAD_DIM, -1)
            _, v_b = torch.split(kv_b, [QK_NOPE_HEAD_DIM, V_HEAD_DIM], dim=1)
            return v_b.half().contiguous().numpy()

        elif entry.kind == "expert_stack":
            # Stack n_experts individual tensors into one 3D tensor
            exp_shape = self._get_shape(entry.sources[0])
            stacked = torch.zeros(
                self.n_experts, exp_shape[0], exp_shape[1],
                dtype=torch.float16,
            )
            for eid, src_name in enumerate(entry.sources):
                exp_data = self._dequant_tensor(src_name)
                stacked[eid] = exp_data.half()
            return stacked.numpy()

        else:
            raise ValueError(f"Unknown tensor kind: {entry.kind!r}")

    # ─── Main conversion ─────────────────────────────────────────────

    def convert(self):
        """Run the full streaming conversion."""
        t_start = time.time()

        # Step 1: Build plan
        logger.info("Step 1/4: Building tensor plan...")
        self._build_plan()

        # Step 2: Create GGUF writer, register tensors, inject metadata
        logger.info("Step 2/4: Registering tensors and metadata...")
        writer_cls = require_gguf_writer()
        writer = writer_cls(str(self.output_path), self.arch)

        # Inject architecture metadata
        inject_mistral_metadata(writer, str(self.config_path), self.arch)

        # Inject tokenizer (Tekken BPE from HF tokenizer files)
        inject_tekken_tokenizer(writer, self.model_dir)

        # Register all tensor info
        for entry in self.plan:
            writer.add_tensor_info(
                entry.gguf_name,
                entry.shape,
                entry.dtype,
                entry.nbytes,
            )
        logger.info(f"  Registered {len(self.plan)} tensors")

        # Step 3: Write header (GGUF magic, KV data, tensor info)
        logger.info("Step 3/4: Writing GGUF header...")
        writer.write_header_to_file(self.output_path)
        writer.write_kv_data_to_file()
        writer.write_ti_data_to_file()

        # Step 4: Stream tensor data
        logger.info("Step 4/4: Streaming tensor data...")
        total_bytes_written = 0
        for i, entry in enumerate(self.plan):
            t0 = time.time()
            data = self._process_tensor(entry)
            dt = time.time() - t0

            writer.write_tensor_data(data)
            total_bytes_written += data.nbytes

            # Progress logging
            if (i + 1) % 10 == 0 or entry.kind == "expert_stack" or i < 5:
                gb_written = total_bytes_written / (1024**3)
                logger.info(
                    f"  [{i+1}/{len(self.plan)}] {entry.gguf_name} "
                    f"({entry.kind}, {data.shape}, {dt:.1f}s) "
                    f"[{gb_written:.1f} GB written]"
                )

        # Cleanup
        self.shard_cache.close_all()

        elapsed = time.time() - t_start
        final_size = os.path.getsize(self.output_path)
        logger.info(
            f"Conversion complete: {final_size / (1024**3):.1f} GB "
            f"in {elapsed:.0f}s ({elapsed/60:.1f} min)"
        )


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Streaming safetensor → GGUF converter for Mistral Large 3 675B",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python3 streaming_gguf_converter.py \\
        --model-dir /path/to/Mistral-Large-Quantization-Project \\
        --output /path/to/mistral-large-3-675b-f16.gguf \\
        --config /path/to/Mistral-Large-Quantization-Project/params.json

Test mode (first 3 layers only):
  python3 streaming_gguf_converter.py \\
        --model-dir /path/to/Mistral-Large-Quantization-Project \\
    --output /tmp/test-3layer.gguf \\
        --config /path/to/Mistral-Large-Quantization-Project/params.json \\
    --max-layers 3
""",
    )
    parser.add_argument("--model-dir", required=True, type=Path,
                        help="Directory with safetensor shards and index.json")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output GGUF file path")
    parser.add_argument("--config", required=True, type=Path,
                        help="Path to params.json (Mistral format)")
    parser.add_argument("--arch", default="deepseek2",
                        help="GGUF architecture (default: deepseek2)")
    parser.add_argument("--max-layers", type=int, default=None,
                        help="Limit conversion to first N layers (for testing)")
    parser.add_argument("--max-shard-handles", type=int, default=8,
                        help="Max simultaneously open shard files")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable debug logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    n_layers = args.max_layers if args.max_layers else N_LAYERS

    converter = StreamingGGUFConverter(
        model_dir=args.model_dir,
        output_path=args.output,
        config_path=args.config,
        arch=args.arch,
        n_layers=n_layers,
        max_shard_handles=args.max_shard_handles,
    )
    converter.convert()


if __name__ == "__main__":
    main()
