#!/usr/bin/env python3
"""Write a decoded TurboQuant shardlet to a GGUF file loadable by llama.cpp.

This is the Phase 1 quantizer hook: it takes the f32 decoded output from the
OpenCL sidecar and writes it as a named GGUF tensor that llama.cpp can load
via its model loader.

Usage:
    python3 gguf_shardlet_writer.py \
        --sidecar-metadata results/ggml_opencl_sidecar_decode.json \
        --tensor-name blk.0.attn_q.weight \
        --output results/shardlet_blk0_attn_q.gguf

The output is a minimal single-tensor GGUF file. It can be merged with other
shardlet GGUFs to build a full model, or loaded individually for verification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

# llama.cpp ships a gguf Python package in gguf-py/
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


PACKAGE_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = PACKAGE_DIR.parent
RESULTS_DIR = EXPERIMENT_DIR / "results"
DEFAULT_SIDECAR_METADATA = RESULTS_DIR / "ggml_opencl_sidecar_decode.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_decoded_f32(metadata_path: Path) -> tuple[np.ndarray, dict]:
    """Load the decoded f32 sidecar output and validate it."""
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not metadata.get("ok", False):
        raise RuntimeError(f"sidecar metadata not ok: {metadata_path}")
    if metadata.get("dtype") != "float32":
        raise RuntimeError(f"expected float32 output, got {metadata.get('dtype')}")

    output_path = Path(metadata["output"]).expanduser()
    if not output_path.is_absolute():
        output_path = (metadata_path.parent / output_path).resolve()
    if not output_path.is_file():
        raise FileNotFoundError(f"sidecar output: {output_path}")

    rows = int(metadata["rows"])
    vector_dim = int(metadata["vector_dim"])
    expected_bytes = rows * vector_dim * 4
    actual_bytes = output_path.stat().st_size
    if actual_bytes != expected_bytes:
        raise RuntimeError(
            f"byte mismatch: file {actual_bytes}, expected {expected_bytes}"
        )

    # Verify hash
    observed_sha = sha256_file(output_path)
    if observed_sha != metadata["output_sha256"]:
        raise RuntimeError(
            f"SHA-256 mismatch: {observed_sha} != {metadata['output_sha256']}"
        )

    data = np.fromfile(output_path, dtype=np.float32).reshape(rows, vector_dim)
    return data, metadata


def write_gguf(
    data: np.ndarray,
    tensor_name: str,
    output_path: Path,
    source_metadata: dict,
) -> dict:
    """Write a single-tensor GGUF file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer_cls = require_gguf_writer()
    writer = writer_cls(output_path, arch="llama")

    # Add source provenance as metadata
    writer.add_description(
        f"TurboQuant q4 shardlet: {source_metadata.get('prefix', 'unknown')}"
    )
    writer.add_file_type(0)  # F32

    # Add the tensor — llama.cpp uses ne[0]=cols, ne[1]=rows (row-major f32)
    writer.add_tensor(tensor_name, data)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    # Verify output
    output_sha = sha256_file(output_path)
    output_bytes = output_path.stat().st_size

    result = {
        "ok": True,
        "tensor_name": tensor_name,
        "shape": list(data.shape),
        "dtype": "float32",
        "data_bytes": data.nbytes,
        "gguf_path": str(output_path),
        "gguf_bytes": output_bytes,
        "gguf_sha256": output_sha,
        "source_prefix": source_metadata.get("prefix", ""),
        "source_package": source_metadata.get("package", ""),
        "source_bridge_url": source_metadata.get("bridge_url", ""),
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sidecar-metadata",
        type=Path,
        default=DEFAULT_SIDECAR_METADATA,
        help="Path to ggml_opencl_sidecar_decode.json",
    )
    parser.add_argument(
        "--tensor-name",
        required=True,
        help="GGUF tensor name (e.g., blk.0.attn_q.weight)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output GGUF file path (default: results/<tensor_name>.gguf)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.output is None:
        safe_name = args.tensor_name.replace(".", "_").replace("/", "_")
        args.output = RESULTS_DIR / f"shardlet_{safe_name}.gguf"

    print(f"loading decoded f32 from {args.sidecar_metadata}")
    data, metadata = load_decoded_f32(args.sidecar_metadata)
    print(f"  shape: {data.shape}, dtype: {data.dtype}, bytes: {data.nbytes}")

    print(f"writing GGUF: {args.output}")
    print(f"  tensor: {args.tensor_name}")
    result = write_gguf(data, args.tensor_name, args.output, metadata)

    # Save result
    result_path = RESULTS_DIR / "gguf_shardlet_writer.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"  gguf bytes: {result['gguf_bytes']}")
    print(f"  gguf sha256: {result['gguf_sha256'][:16]}...")
    print(f"  result: {result_path}")
    print("done")


if __name__ == "__main__":
    main()
