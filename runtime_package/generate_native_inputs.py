#!/usr/bin/env python3
"""Generate small native runtime inputs shared by Python and C runners."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
MODEL_LAB = Path(__file__).resolve().parents[3]
TURBOQUANT_ROOT = Path(
    os.environ.get("TURBOQUANT_ROOT", str(MODEL_LAB / "external/0xSero_turboquant"))
)
OUTPUT_DIR = PACKAGE_DIR / "generated_inputs"
VECTOR_DIM = 128
MSE_BITS = 3
QJL_SEED = 1042

sys.path.insert(0, str(TURBOQUANT_ROOT))
np = importlib.import_module("numpy")


def load_turboquant_helpers() -> tuple[object, object, object]:
    sys.path.insert(0, str(TURBOQUANT_ROOT))
    torch = importlib.import_module("torch")
    get_codebook = importlib.import_module("turboquant.codebook").get_codebook
    generate_qjl_matrix = importlib.import_module(
        "turboquant.rotation"
    ).generate_qjl_matrix
    return torch, get_codebook, generate_qjl_matrix


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_raw(path: Path, array: object) -> dict[str, object]:
    payload = np.ascontiguousarray(array, dtype=np.float32).tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "path": str(path),
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
    }


def main() -> None:
    torch, get_codebook, generate_qjl_matrix = load_turboquant_helpers()
    centroids = np.asarray(
        get_codebook(VECTOR_DIM, MSE_BITS)["centroids"],
        dtype=np.float32,
    )
    qjl_matrix = generate_qjl_matrix(
        VECTOR_DIM,
        torch.device("cpu"),
        torch.float32,
        seed=QJL_SEED,
    ).numpy()

    manifest = {
        "schema_version": 1,
        "vector_dim": VECTOR_DIM,
        "mse_bits": MSE_BITS,
        "qjl_seed": QJL_SEED,
        "centroids": write_raw(
            OUTPUT_DIR / "centroids_d128_b3.raw",
            centroids,
        )
        | {"shape": [int(centroids.shape[0])], "dtype": "float32"},
        "qjl_matrix": write_raw(
            OUTPUT_DIR / "qjl_matrix_d128_seed1042.raw",
            qjl_matrix,
        )
        | {"shape": [VECTOR_DIM, VECTOR_DIM], "dtype": "float32"},
    }
    manifest_path = OUTPUT_DIR / "generated_inputs_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()