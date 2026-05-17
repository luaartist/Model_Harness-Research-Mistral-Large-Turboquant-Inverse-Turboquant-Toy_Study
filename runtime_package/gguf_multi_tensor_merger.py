#!/usr/bin/env python3
"""Merge multiple single-tensor GGUF shardlet files into one multi-tensor GGUF.

This assembles decoded TurboQuant shardlets into a GGUF file that llama.cpp
can load as a (partial) model. Each shardlet is a single-tensor GGUF produced
by gguf_shardlet_writer.py.

Architecture:
    shardlet_blk_0_attn_q_a_weight.gguf  ─┐
    shardlet_blk_0_attn_q_b_weight.gguf  ─┤
    shardlet_blk_0_ffn_gate_weight.gguf  ─┼──▶  merged_model.gguf
    ...                                   ─┘

Usage:
    # Merge all shardlet_*.gguf in results/ into one file
    python3 gguf_multi_tensor_merger.py

    # Merge specific files
    python3 gguf_multi_tensor_merger.py \
        --shardlets results/shardlet_blk_0_*.gguf \
        --output results/merged_layer0.gguf

    # Merge with model metadata (for llama.cpp loading)
    python3 gguf_multi_tensor_merger.py \
        --arch llama \
        --model-name "Mistral-Large-3-675B-TurboQuant-Q4" \
        --output results/merged_model.gguf
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

GGUF_PY = Path(os.environ.get("GGUF_PY_PATH", "third_party/llama.cpp/gguf-py"))
sys.path.insert(0, str(GGUF_PY))

try:
    from gguf import GGUFReader, GGUFWriter  # noqa: E402
except ImportError:  # pragma: no cover - optional llama.cpp dependency
    GGUFReader = None  # type: ignore[assignment]
    GGUFWriter = None  # type: ignore[assignment]

from gguf_metadata_injector import inject_mistral_metadata  # noqa: E402


PACKAGE_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = PACKAGE_DIR.parent
RESULTS_DIR = EXPERIMENT_DIR / "results"


def require_gguf() -> tuple[object, object]:
    if GGUFReader is None or GGUFWriter is None:
        raise RuntimeError(
            "gguf is not importable. Set GGUF_PY_PATH to llama.cpp/gguf-py "
            "or install the llama.cpp gguf Python package."
        )
    return GGUFReader, GGUFWriter


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def discover_shardlets(directory: Path) -> list[Path]:
    """Find all shardlet_*.gguf files in directory, sorted by name."""
    pattern = str(directory / "shardlet_*.gguf")
    paths = sorted(Path(p) for p in glob.glob(pattern))
    return paths


def read_shardlet(path: Path) -> dict[str, Any]:
    """Read a single-tensor shardlet GGUF and extract tensor info."""
    reader_cls, _ = require_gguf()
    reader = reader_cls(path)
    if len(reader.tensors) == 0:
        raise RuntimeError(f"no tensors in {path}")
    if len(reader.tensors) > 1:
        raise RuntimeError(f"expected 1 tensor in shardlet, got {len(reader.tensors)}: {path}")

    t = reader.tensors[0]
    # Map numpy dtype from tensor type
    dtype_map = {
        "F32": np.float32,
        "F16": np.float16,
        "BF16": np.float16,  # fallback — bf16 needs special handling
    }
    dtype = dtype_map.get(t.tensor_type.name, np.float32)
    data = np.frombuffer(t.data, dtype=dtype).reshape(t.shape)

    return {
        "name": str(t.name),
        "shape": list(int(x) for x in t.shape),
        "dtype": t.tensor_type.name,
        "data": data,
        "source_path": str(path),
        "source_bytes": path.stat().st_size,
    }


def merge_shardlets(
    shardlet_paths: list[Path],
    output_path: Path,
    arch: str = "deepseek2",
    model_name: str | None = None,
    description: str | None = None,
    config_path: Path | str | None = None,
) -> dict[str, Any]:
    """Merge multiple single-tensor shardlet GGUFs into one multi-tensor GGUF.

    Args:
        shardlet_paths: Paths to single-tensor GGUF files
        output_path: Where to write the merged GGUF
        arch: Model architecture name (default: deepseek2 for MLA models)
        model_name: Optional model name metadata
        description: Optional description metadata
        config_path: Path to Mistral params.json for full metadata injection.
                     If provided, uses GGUFMetadataInjector (~30 KV pairs).
                     If None, falls back to minimal hardcoded metadata.

    Returns:
        Result dictionary with merge statistics
    """
    if not shardlet_paths:
        raise RuntimeError("no shardlet files to merge")

    t0 = time.monotonic()

    # Read all shardlets
    tensors: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    total_data_bytes = 0

    for path in shardlet_paths:
        info = read_shardlet(path)
        if info["name"] in seen_names:
            raise RuntimeError(f"duplicate tensor name: {info['name']} (from {path})")
        seen_names.add(info["name"])
        tensors.append(info)
        total_data_bytes += info["data"].nbytes

    # Sort tensors by name for deterministic output
    tensors.sort(key=lambda t: t["name"])

    # Write merged GGUF
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _, writer_cls = require_gguf()
    writer = writer_cls(output_path, arch=arch)

    injection_audit = None
    if config_path is not None:
        # Full metadata injection via DynamicFlux-pattern injector
        injection_audit = inject_mistral_metadata(writer, config_path, arch=arch)
    else:
        # Minimal fallback (no params.json available)
        if model_name:
            writer.add_name(model_name)
        if description:
            writer.add_description(description)
        else:
            writer.add_description(
                f"TurboQuant q4 merged model: {len(tensors)} tensors from shardlets"
            )
        writer.add_file_type(0)  # F32

    # Add all tensors
    for t in tensors:
        writer.add_tensor(t["name"], t["data"])

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    duration = round(time.monotonic() - t0, 3)
    output_sha = sha256_file(output_path)
    output_bytes = output_path.stat().st_size

    result = {
        "ok": True,
        "output_path": str(output_path),
        "output_bytes": output_bytes,
        "output_sha256": output_sha,
        "arch": arch,
        "model_name": model_name,
        "tensor_count": len(tensors),
        "total_data_bytes": total_data_bytes,
        "duration_seconds": duration,
        "metadata_injected": injection_audit is not None,
        "metadata_kv_pairs": injection_audit.total_kv_pairs if injection_audit else 0,
        "tensors": [
            {
                "name": t["name"],
                "shape": t["shape"],
                "dtype": t["dtype"],
                "data_bytes": t["data"].nbytes,
                "source": t["source_path"],
            }
            for t in tensors
        ],
    }
    if injection_audit:
        result["injection_audit"] = injection_audit.to_dict()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--shardlets",
        nargs="*",
        type=Path,
        default=None,
        help="Paths to shardlet GGUF files (default: auto-discover in results/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "merged_model.gguf",
        help="Output merged GGUF path",
    )
    parser.add_argument(
        "--arch",
        default="deepseek2",
        help="Model architecture (default: deepseek2 for MLA models)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to Mistral params.json for full metadata injection",
    )
    parser.add_argument(
        "--model-name",
        default="Mistral-Large-3-675B-TurboQuant-Q4",
        help="Model name metadata",
    )
    parser.add_argument(
        "--description",
        default=None,
        help="Description metadata",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.shardlets:
        shardlet_paths = sorted(args.shardlets)
    else:
        shardlet_paths = discover_shardlets(RESULTS_DIR)

    if not shardlet_paths:
        print("error: no shardlet files found")
        print(f"  searched: {RESULTS_DIR}/shardlet_*.gguf")
        raise SystemExit(1)

    print(f"merging {len(shardlet_paths)} shardlets -> {args.output}")
    for p in shardlet_paths:
        print(f"  {p.name}")

    result = merge_shardlets(
        shardlet_paths,
        args.output,
        arch=args.arch,
        model_name=args.model_name,
        description=args.description,
        config_path=args.config,
    )

    # Save result
    result_path = RESULTS_DIR / "gguf_merger_result.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"\nmerged GGUF:")
    print(f"  tensors:    {result['tensor_count']}")
    print(f"  data bytes: {result['total_data_bytes']:,}")
    print(f"  gguf bytes: {result['output_bytes']:,}")
    print(f"  sha256:     {result['output_sha256'][:16]}...")
    print(f"  duration:   {result['duration_seconds']}s")
    print(f"  result:     {result_path}")
    print("done")


if __name__ == "__main__":
    main()
