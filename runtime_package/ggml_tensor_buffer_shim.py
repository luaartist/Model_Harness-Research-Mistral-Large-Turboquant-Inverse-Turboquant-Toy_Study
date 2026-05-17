#!/usr/bin/env python3
"""Build a GGML-style tensor descriptor from sidecar decoded f32 bytes.

This is still an external shim, not a llama.cpp core patch. It validates the
sidecar output file, maps the row-major f32 block to GGML dimensions and byte
strides, and writes a descriptor that a native loader can implement directly.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
from typing import Any


PACKAGE_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = PACKAGE_DIR.parent
RESULTS_DIR = EXPERIMENT_DIR / "results"
DEFAULT_SIDECAR_METADATA = RESULTS_DIR / "ggml_opencl_sidecar_decode.json"
DEFAULT_DESCRIPTOR = RESULTS_DIR / "ggml_tensor_buffer_shim.json"

np = importlib.import_module("numpy")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def resolve_output_path(raw_path: str, metadata_path: Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (metadata_path.parent / path).resolve()


def validate_sidecar(
    metadata_path: Path,
    tolerance: float,
) -> tuple[dict[str, Any], Path, str, list[float]]:
    metadata = load_json(metadata_path)
    if not metadata.get("ok", False):
        raise RuntimeError(f"sidecar metadata is not ok: {metadata_path}")
    if metadata.get("dtype") != "float32":
        raise RuntimeError(f"expected float32 sidecar output: {metadata_path}")

    output_path = resolve_output_path(str(metadata["output"]), metadata_path)
    if not output_path.is_file():
        raise FileNotFoundError(output_path)

    rows = int(metadata["rows"])
    vector_dim = int(metadata["vector_dim"])
    expected_bytes = rows * vector_dim * 4
    observed_bytes = output_path.stat().st_size
    if observed_bytes != expected_bytes:
        raise RuntimeError(
            f"output byte mismatch: observed {observed_bytes}, "
            f"expected {expected_bytes}"
        )
    if observed_bytes != int(metadata["output_bytes"]):
        raise RuntimeError(
            f"metadata byte mismatch: file {observed_bytes}, "
            f"metadata {metadata['output_bytes']}"
        )

    observed_sha256 = sha256_file(output_path)
    if observed_sha256 != metadata["output_sha256"]:
        raise RuntimeError(
            f"output sha256 mismatch: observed {observed_sha256}, "
            f"expected {metadata['output_sha256']}"
        )

    mapped = np.memmap(
        output_path,
        mode="r",
        dtype=np.float32,
        shape=(rows, vector_dim),
    )
    sample = np.asarray(mapped.reshape(-1)[:8], dtype=np.float32).tolist()
    expected_sample = metadata.get("sample", [])
    if expected_sample:
        expected = np.asarray(expected_sample[: len(sample)], dtype=np.float32)
        observed = np.asarray(sample, dtype=np.float32)
        max_abs = float(np.max(np.abs(observed - expected)))
        if max_abs > tolerance:
            raise RuntimeError(
                f"sample mismatch: max_abs {max_abs} > tolerance {tolerance}"
            )
    return metadata, output_path, observed_sha256, sample


def build_descriptor(
    metadata_path: Path,
    tensor_name: str,
    tolerance: float,
) -> dict[str, Any]:
    metadata, output_path, output_sha256, sample = validate_sidecar(
        metadata_path,
        tolerance,
    )
    rows = int(metadata["rows"])
    vector_dim = int(metadata["vector_dim"])
    item_size = 4
    row_bytes = vector_dim * item_size
    total_bytes = rows * row_bytes
    name = tensor_name or f"{metadata['prefix']}.decoded_f32"

    return {
        "ok": True,
        "adapter": "ggml-tensor-buffer-shim",
        "source_metadata": str(metadata_path),
        "source_output": str(output_path),
        "source_output_sha256": output_sha256,
        "tensor": {
            "name": name,
            "ggml_type": "GGML_TYPE_F32",
            "dtype": "float32",
            "shape_row_major": [rows, vector_dim],
            "ggml_ne": [vector_dim, rows, 1, 1],
            "ggml_nb": [item_size, row_bytes, total_bytes, total_bytes],
            "n_dims": 2,
            "nbytes": total_bytes,
            "data_offset_bytes": 0,
            "layout": "row-major f32; GGML ne[0]=vector_dim, ne[1]=rows",
        },
        "loader_contract": {
            "allocation": (
                "ggml_new_tensor_2d(ctx, GGML_TYPE_F32, vector_dim, rows)"
            ),
            "copy": "memcpy(tensor->data, mapped_f32_bytes, nbytes)",
            "transpose_required": False,
            "backend_upload": (
                "backend buffers may upload the same contiguous f32 block"
            ),
        },
        "source_package": {
            "package": metadata["package"],
            "prefix": metadata["prefix"],
            "manifest": metadata["manifest"],
            "bridge_url": metadata["bridge_url"],
            "bridge_status": metadata["bridge_status"],
        },
        "sample": sample,
        "claim_boundary": metadata.get("claim_boundary", ""),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sidecar-metadata",
        type=Path,
        default=DEFAULT_SIDECAR_METADATA,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_DESCRIPTOR)
    parser.add_argument("--tensor-name", default="")
    parser.add_argument("--sample-tolerance", type=float, default=1e-7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    descriptor = build_descriptor(
        args.sidecar_metadata,
        args.tensor_name,
        args.sample_tolerance,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(descriptor, indent=2), encoding="utf-8")
    tensor = descriptor["tensor"]
    print("ggml tensor buffer shim complete")
    print(f"  tensor: {tensor['name']}")
    print(f"  ggml_type: {tensor['ggml_type']}")
    print(f"  ggml_ne: {tensor['ggml_ne']}")
    print(f"  ggml_nb: {tensor['ggml_nb']}")
    print(f"  nbytes: {tensor['nbytes']}")
    print(f"  descriptor: {args.output}")


if __name__ == "__main__":
    main()
