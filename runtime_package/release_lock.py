#!/usr/bin/env python3
"""Write or check the GGML loader-surface shardlet release hash lock."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PACKAGE_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = PACKAGE_DIR.parent
RESULTS_DIR = EXPERIMENT_DIR / "results"
REPO_ROOT = PACKAGE_DIR.parent
RELEASE_TAG = "v0.3-ggml-loader-surface"
LOCK_PATH = PACKAGE_DIR / f"RELEASE_LOCK.{RELEASE_TAG}.json"


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


def resolve_package_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def hash_record(label: str, path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label}: {path}")
    return {
        "label": label,
        "path": display_path(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def manifest_artifact_record(manifest: dict[str, Any]) -> dict[str, Any]:
    artifact = manifest["artifact"]
    path = resolve_package_path(artifact["path"])
    if path.is_file():
        return hash_record("artifact", path)
    return {
        "label": "artifact",
        "path": artifact["path"],
        "bytes": artifact["bytes"],
        "sha256": artifact["sha256"],
        "external": True,
    }


def build_lock() -> dict[str, Any]:
    manifest_path = PACKAGE_DIR / "runtime_package_manifest.json"
    manifest = load_json(manifest_path)
    records = [
        manifest_artifact_record(manifest),
        hash_record("kernel", resolve_package_path(manifest["kernel"]["path"])),
        hash_record("manifest", manifest_path),
        hash_record(
            "native_centroids",
            resolve_package_path(manifest["generated_inputs"]["centroids"]["path"]),
        ),
        hash_record(
            "native_qjl_matrix",
            resolve_package_path(manifest["generated_inputs"]["qjl_matrix"]["path"]),
        ),
        hash_record(
            "generated_inputs_manifest",
            PACKAGE_DIR / "generated_inputs/generated_inputs_manifest.json",
        ),
        hash_record(
            "packaging_boundary",
            PACKAGE_DIR / "PACKAGING_BOUNDARY.json",
        ),
        hash_record(
            "model_profile_template",
            PACKAGE_DIR / "MODEL_PROFILE_TEMPLATE.json",
        ),
        hash_record("release_notes", PACKAGE_DIR / "RELEASE_NOTES.md"),
        hash_record("runner_readme", PACKAGE_DIR / "RUNNER_README.md"),
        hash_record("package_boundary", PACKAGE_DIR / "PACKAGE_BOUNDARY.md"),
        hash_record("package_catalog", PACKAGE_DIR / "package_catalog.json"),
    ]
    return {
        "schema_version": 1,
        "release_tag": RELEASE_TAG,
        "package": manifest["package"],
        "prefix": manifest["prefix"],
        "verify_command": "python release_lock.py --check",
        "hashes": records,
        "implemented_backends": {
            "decoded_f32_file": "maps predecoded sidecar output",
            "http_bridge": "native C decode through the HIP/OpenCL HTTP bridge",
            "ggml_loader_surface": "materializes decoded output into a real ggml_tensor",
        },
        "reserved_backends": {
            "inprocess_opencl_backend": "reserved, not implemented",
        },
        "out_of_scope": {
            "full_model_inference": f"not part of {RELEASE_TAG}",
            "native_gguf_quant_type": f"not part of {RELEASE_TAG}",
        },
        "claim_boundary": manifest["claim_boundary"],
    }


def compare_locks(
    expected: dict[str, Any],
    observed: dict[str, Any],
) -> list[str]:
    failures: list[str] = []
    expected_records = {
        record["label"]: record for record in expected["hashes"]
    }
    observed_records = {
        record["label"]: record for record in observed["hashes"]
    }
    for label, expected_record in sorted(expected_records.items()):
        observed_record = observed_records.get(label)
        if observed_record is None:
            failures.append(f"missing observed hash record: {label}")
            continue
        for key in ("path", "bytes", "sha256"):
            if observed_record[key] != expected_record[key]:
                failures.append(
                    f"{label}.{key}: observed {observed_record[key]!r}, "
                    f"expected {expected_record[key]!r}"
                )
    for label in sorted(set(observed_records) - set(expected_records)):
        failures.append(f"unexpected observed hash record: {label}")
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    observed = build_lock()
    if args.write:
        LOCK_PATH.write_text(
            json.dumps(observed, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"wrote release lock: {LOCK_PATH}")
        return

    expected = load_json(LOCK_PATH)
    failures = compare_locks(expected, observed)
    print(f"release_tag: {observed['release_tag']}")
    print(f"hash_records: {len(observed['hashes'])}")
    print(f"valid: {not failures}")
    for failure in failures:
        print(f"failure: {failure}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
