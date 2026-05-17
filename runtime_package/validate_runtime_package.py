#!/usr/bin/env python3
"""Validate the runner-facing runtime package manifest."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any


PACKAGE_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = PACKAGE_DIR.parent
REPO_ROOT = PACKAGE_DIR.parent
MODEL_LAB = Path(__file__).resolve().parents[3]
TURBOQUANT_ROOT = Path(
    os.environ.get("TURBOQUANT_ROOT", str(MODEL_LAB / "external/0xSero_turboquant"))
)
RESULTS_DIR = EXPERIMENT_DIR / "results"

sys.path.insert(0, str(TURBOQUANT_ROOT))
np = importlib.import_module("numpy")
safe_open = importlib.import_module("safetensors").safe_open


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_manifest_path(value: str | Path) -> Path:
    """Resolve package manifest paths from repo root unless already absolute."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def array_hash(array: np.ndarray) -> str:
    return sha256_bytes(np.ascontiguousarray(array).tobytes())


def record_failure(failures: list[str], message: str) -> None:
    failures.append(message)


def check_equal(
    failures: list[str],
    label: str,
    observed: Any,
    expected: Any,
) -> None:
    if observed != expected:
        record_failure(
            failures,
            f"{label}: observed {observed!r}, expected {expected!r}",
        )


def validate_tensor(
    failures: list[str],
    handle: Any,
    key: str,
    record: dict[str, Any],
) -> None:
    tensor = handle.get_tensor(record["name"]).numpy()
    target_dtype = np.dtype(record["dtype"])
    array = np.ascontiguousarray(tensor.astype(target_dtype, copy=False))
    check_equal(failures, f"{key}.shape", list(array.shape), record["shape"])
    check_equal(failures, f"{key}.dtype", str(array.dtype), record["dtype"])
    check_equal(failures, f"{key}.bytes", int(array.nbytes), record["bytes"])
    check_equal(failures, f"{key}.sha256", array_hash(array), record["sha256"])


def validate_generated_inputs(
    failures: list[str],
    manifest: dict[str, Any],
) -> None:
    generated = manifest["generated_inputs"]
    for key in ("centroids", "qjl_matrix"):
        record = generated[key]
        input_path = resolve_manifest_path(record["path"])
        if not input_path.is_file():
            record_failure(failures, f"{key}.path missing: {input_path}")
            continue
        array = np.fromfile(input_path, dtype=np.dtype(record["dtype"]))
        array = np.ascontiguousarray(array.reshape(record["shape"]))
        check_equal(
            failures,
            f"{key}.shape",
            list(array.shape),
            record["shape"],
        )
        check_equal(
            failures,
            f"{key}.dtype",
            str(array.dtype),
            record["dtype"],
        )
        check_equal(
            failures,
            f"{key}.bytes",
            int(array.nbytes),
            record["bytes"],
        )
        check_equal(
            failures,
            f"{key}.sha256",
            array_hash(array),
            record["sha256"],
        )


def validate_manifest(
    manifest_path: Path,
    output_path: Path | None = RESULTS_DIR / "runtime_package_validation.json",
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    artifact_path = resolve_manifest_path(manifest["artifact"]["path"])
    kernel_path = resolve_manifest_path(manifest["kernel"]["path"])

    if not artifact_path.is_file():
        raise FileNotFoundError(
            "artifact not found: "
            f"{artifact_path}. Provide the hash-pinned shardlet artifact "
            "or override the manifest path before full validation."
        )
    if not kernel_path.is_file():
        raise FileNotFoundError(f"kernel not found: {kernel_path}")

    check_equal(failures, "schema_version", manifest["schema_version"], 1)
    check_equal(
        failures,
        "artifact.bytes",
        artifact_path.stat().st_size,
        manifest["artifact"]["bytes"],
    )
    check_equal(
        failures,
        "artifact.sha256",
        sha256_file(artifact_path),
        manifest["artifact"]["sha256"],
    )
    check_equal(
        failures,
        "kernel.bytes",
        kernel_path.stat().st_size,
        manifest["kernel"]["bytes"],
    )
    check_equal(
        failures,
        "kernel.sha256",
        sha256_file(kernel_path),
        manifest["kernel"]["sha256"],
    )

    with safe_open(str(artifact_path), framework="pt", device="cpu") as handle:
        for key, record in manifest["tensors"].items():
            validate_tensor(failures, handle, key, record)

    validate_generated_inputs(failures, manifest)

    result = {
        "manifest": str(manifest_path),
        "package": manifest["package"],
        "prefix": manifest["prefix"],
        "artifact": str(artifact_path),
        "kernel": str(kernel_path),
        "tensor_count": len(manifest["tensors"]),
        "generated_input_count": len(manifest["generated_inputs"]),
        "stored_tensor_bytes": manifest["byte_accounting"][
            "stored_tensor_bytes"
        ],
        "generated_matrix_bytes": manifest["byte_accounting"][
            "generated_matrix_bytes"
        ],
        "valid": not failures,
        "failures": failures,
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PACKAGE_DIR / "runtime_package_manifest.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "runtime_package_validation.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = validate_manifest(args.manifest, args.output)
    print(f"package: {result['package']}")
    print(f"prefix: {result['prefix']}")
    print(f"tensor_count: {result['tensor_count']}")
    print(f"generated_input_count: {result['generated_input_count']}")
    print(f"valid: {result['valid']}")
    if result["failures"]:
        for failure in result["failures"]:
            print(f"failure: {failure}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
