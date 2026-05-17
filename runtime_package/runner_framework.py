#!/usr/bin/env python3
"""First-build orchestration notes for the shardlet runtime package.

Python is the right place to keep this while the package contract is moving. It
can call the validator, sidecar descriptor, and C runner shim while leaving a
readable map for the later Rust port. The notes below are deliberately
framework concepts, not claims that the Rust or native bridge backend exists.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PACKAGE_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = PACKAGE_DIR.parent
RESULTS_DIR = EXPERIMENT_DIR / "results"
CATALOG_RESULTS_DIR = RESULTS_DIR / "catalog_runs"
PRIVATE_RUNNER_DIR = PACKAGE_DIR / "private_runner_shim"
DEFAULT_MANIFEST = PACKAGE_DIR / "runtime_package_manifest.json"
DEFAULT_CATALOG = RESULTS_DIR / "package_catalog.json"
DEFAULT_CATALOG_REPORT = RESULTS_DIR / "catalog_run_report.json"
DEFAULT_NATIVE_CENTROIDS = PACKAGE_DIR / "generated_inputs/centroids_d128_b3.raw"
DEFAULT_NATIVE_QJL_MATRIX = PACKAGE_DIR / "generated_inputs/qjl_matrix_d128_seed1042.raw"
DEFAULT_AUDIT = RESULTS_DIR / "python_runner_framework_audit.json"
DEFAULT_SIDECAR_METADATA = RESULTS_DIR / "ggml_opencl_sidecar_decode.json"
DEFAULT_F32 = RESULTS_DIR / "ggml_opencl_sidecar_decode.f32"
DEFAULT_DESCRIPTOR = RESULTS_DIR / "ggml_tensor_buffer_shim.json"
DEFAULT_RUNNER_AUDIT = RESULTS_DIR / "private_runner_api_audit.json"
DEFAULT_HTTP_BRIDGE_AUDIT = RESULTS_DIR / "private_runner_http_bridge_audit.json"
DEFAULT_HTTP_BRIDGE_F32 = RESULTS_DIR / "private_runner_http_bridge_decode.f32"
DEFAULT_LOADER_AUDIT = RESULTS_DIR / "ggml_loader_surface_audit.json"


FRAMEWORK_NOTES: list[dict[str, str]] = [
    {
        "topic": "python_first_build",
        "note": (
            "Use Python as the mutable orchestration layer: validate hashes, "
            "call existing package scripts, collect audits, and keep behavior "
            "easy to inspect while the artifact format is still changing."
        ),
    },
    {
        "topic": "gate_boundary",
        "note": (
            "The gate is manifest plus SHA-256 plus tensor descriptor plus "
            "native audit. Runners should refuse stale package inputs before "
            "decode or tensor mapping."
        ),
    },
    {
        "topic": "rust_port_shape",
        "note": (
            "Later Rust should mirror this as Result<TensorBuffer, Error>, "
            "a DecodeBackend trait, serde manifests, sha2 verification, and "
            "explicit unsupported backend errors until each backend is real."
        ),
    },
    {
        "topic": "backend_trait_sketch",
        "note": (
            "trait DecodeBackend { fn decode(&self, package: &Package) "
            "-> Result<TensorBuffer>; } with file, http_bridge, and "
            "inprocess_opencl implementations added one at a time."
        ),
    },
    {
        "topic": "claim_boundary",
        "note": (
            "This framework proves orchestration and one shardlet handoff. It "
            "does not prove full Mistral inference, a native GGUF type, or "
            "market compression superiority."
        ),
    },
]


@dataclass(frozen=True)
class CommandResult:
    label: str
    argv: list[str]
    cwd: Path
    returncode: int
    duration_seconds: float
    stdout_tail: str
    stderr_tail: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def to_json(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "argv": self.argv,
            "cwd": str(self.cwd),
            "returncode": self.returncode,
            "ok": self.ok,
            "duration_seconds": self.duration_seconds,
            "stdout_tail": self.stdout_tail,
            "stderr_tail": self.stderr_tail,
        }


def tail_text(value: str, max_chars: int = 4000) -> str:
    if len(value) <= max_chars:
        return value
    return value[-max_chars:]


def run_command(label: str, argv: list[str], cwd: Path) -> CommandResult:
    started = time.monotonic()
    completed = subprocess.run(
        argv,
        cwd=str(cwd),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return CommandResult(
        label=label,
        argv=argv,
        cwd=cwd,
        returncode=completed.returncode,
        duration_seconds=round(time.monotonic() - started, 6),
        stdout_tail=tail_text(completed.stdout),
        stderr_tail=tail_text(completed.stderr),
    )


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def require_success(result: CommandResult) -> None:
    if result.returncode != 0:
        raise RuntimeError(
            f"{result.label} failed with exit {result.returncode}: "
            f"{result.stderr_tail or result.stdout_tail}"
        )


def run_python_stage(script: str, *args: str) -> CommandResult:
    return run_command(script, [sys.executable, script, *args], PACKAGE_DIR)


def validate_http_boundary(result: CommandResult) -> dict[str, Any]:
    expected = "http_bridge status failed"
    observed = result.stderr_tail + result.stdout_tail
    ok = result.returncode == 1 and expected in observed
    return {
        "ok": ok,
        "expected_exit": 1,
        "observed_exit": result.returncode,
        "expected_message_fragment": expected,
        "observed_tail": tail_text(observed, 1000),
    }


def probe_bridge_status(
    bridge_url: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    status_url = f"{bridge_url.rstrip('/')}/status"
    started = time.monotonic()
    try:
        with urllib.request.urlopen(  # noqa: S310 - local lab bridge probe
            status_url,
            timeout=timeout_seconds,
        ) as response:
            payload = response.read()
            status = json.loads(payload.decode("utf-8"))
            if not isinstance(status, dict):
                raise ValueError("status response was not a JSON object")
            memory_handles = status.get("memory_handles")
            vram_allocated = status.get("vram_allocated")
            return {
                "ok": True,
                "url": status_url,
                "duration_seconds": round(time.monotonic() - started, 6),
                "status": status,
                "memory_clean": memory_handles in (0, None)
                and vram_allocated in (0, None),
            }
    except (
        OSError,
        urllib.error.URLError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        return {
            "ok": False,
            "url": status_url,
            "duration_seconds": round(time.monotonic() - started, 6),
            "error": str(exc),
            "memory_clean": False,
        }


def build_framework_audit(args: argparse.Namespace) -> dict[str, Any]:
    commands: list[CommandResult] = []
    manifest = load_json(args.manifest)
    live_bridge: dict[str, Any] | str = "not checked"

    if args.probe_live_bridge:
        live_bridge = probe_bridge_status(
            str(manifest["runtime"]["bridge_url"]),
            args.bridge_timeout,
        )
        if args.require_live_bridge and not live_bridge["ok"]:
            raise RuntimeError(f"live bridge probe failed: {live_bridge}")

    if args.validate_package:
        package_result = run_python_stage("validate_runtime_package.py")
        commands.append(package_result)
        require_success(package_result)

    descriptor_result = run_python_stage(
        "ggml_tensor_buffer_shim.py",
        "--sidecar-metadata",
        str(args.sidecar_metadata),
        "--output",
        str(args.descriptor),
    )
    commands.append(descriptor_result)
    require_success(descriptor_result)

    make_clean = run_command(
        "make clean",
        ["make", "clean"],
        PRIVATE_RUNNER_DIR,
    )
    commands.append(make_clean)
    require_success(make_clean)

    make_build = run_command("make", ["make"], PRIVATE_RUNNER_DIR)
    commands.append(make_build)
    require_success(make_build)

    runner_result = run_command(
        "private runner decoded_f32_file",
        [
            "./smoke_ggml_tq_tensor_shim",
            str(args.decoded_f32),
            str(args.runner_audit),
            "decoded_f32_file",
        ],
        PRIVATE_RUNNER_DIR,
    )
    commands.append(runner_result)
    require_success(runner_result)

    http_boundary: dict[str, Any] | None = None
    if args.check_backend_boundaries:
        bridge_ok = isinstance(live_bridge, dict) and live_bridge.get("ok")
        if bridge_ok:
            http_result = run_command(
                "private runner http_bridge live decode",
                [
                    "./smoke_ggml_tq_tensor_shim",
                    str(args.decoded_f32),
                    str(DEFAULT_HTTP_BRIDGE_AUDIT),
                    "http_bridge",
                    str(args.manifest),
                    str(manifest["runtime"]["bridge_url"]),
                    str(DEFAULT_HTTP_BRIDGE_F32),
                ],
                PRIVATE_RUNNER_DIR,
            )
            commands.append(http_result)
            require_success(http_result)
            sidecar_sha256 = sha256_file(args.decoded_f32)
            native_sha256 = sha256_file(DEFAULT_HTTP_BRIDGE_F32)
            identity_ok = native_sha256 == sidecar_sha256
            if not identity_ok:
                raise RuntimeError(
                    "http_bridge native output does not match Python sidecar: "
                    f"native={native_sha256} sidecar={sidecar_sha256}"
                )
            http_boundary = {
                "ok": True,
                "implemented": True,
                "mode": "live_decode",
                "audit": str(DEFAULT_HTTP_BRIDGE_AUDIT),
                "identity_check": {
                    "ok": identity_ok,
                    "native_output_path": str(DEFAULT_HTTP_BRIDGE_F32),
                    "native_output_sha256": native_sha256,
                    "python_sidecar_path": str(args.decoded_f32),
                    "python_sidecar_sha256": sidecar_sha256,
                },
            }
        else:
            http_boundary = {
                "ok": True,
                "implemented": True,
                "mode": "skipped_bridge_offline",
                "reason": live_bridge if isinstance(live_bridge, dict) else "not checked",
            }

    package_validation = load_json(
        RESULTS_DIR / "runtime_package_validation.json"
    )
    descriptor = load_json(args.descriptor)
    runner_audit = load_json(args.runner_audit)
    loader_audit = load_json(args.loader_audit)

    command_records = [command.to_json() for command in commands]
    if args.stable_audit:
        for command_record in command_records:
            command_record["duration_seconds"] = 0.0
        if isinstance(live_bridge, dict):
            live_bridge = dict(live_bridge)
            live_bridge["duration_seconds"] = 0.0

    tensor = descriptor["tensor"]
    audit = {
        "ok": True,
        "adapter": "python-runner-framework",
        "role": "first-build orchestrator and framework notes",
        "package": package_validation,
        "descriptor": {
            "path": str(args.descriptor),
            "source_output_sha256": descriptor["source_output_sha256"],
            "tensor": tensor,
        },
        "native_runner": {
            "path": str(args.runner_audit),
            "backend": runner_audit["backend"],
            "tensor": runner_audit["tensor"],
            "loader_contract": runner_audit["loader_contract"],
        },
        "ggml_loader_surface": {
            "path": str(args.loader_audit),
            "ok": loader_audit["ok"],
            "adapter": loader_audit["adapter"],
            "source_backend": loader_audit["source_backend"],
            "ggml": loader_audit["ggml"],
            "copy_check": loader_audit["copy_check"],
            "loader_contract": loader_audit["loader_contract"],
        },
        "live_bridge": live_bridge,
        "backend_boundaries": {
            "decoded_f32_file": "implemented",
            "http_bridge": http_boundary or "not checked",
            "inprocess_opencl": "reserved, not implemented",
        },
        "framework_notes": FRAMEWORK_NOTES,
        "commands": command_records,
    }
    if args.stable_audit:
        for command in audit["commands"]:
            command["duration_seconds"] = 0.0
        live = audit["live_bridge"]
        if isinstance(live, dict):
            live["duration_seconds"] = 0.0
    return audit


# ---- catalog mode --------------------------------------------------------


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def catalog_tensor_records(
    artifact_path: Path,
    prefix: str,
    rows: int,
    vector_dim: int,
    mse_cols: int,
    qjl_cols: int,
) -> dict[str, dict[str, Any]]:
    """Build validator-compatible tensor records for a catalog shardlet."""
    np = importlib.import_module("numpy")
    safe_open = importlib.import_module("safetensors").safe_open
    specs: dict[str, tuple[str, list[int], str]] = {
        "mse_indices": (f"{prefix}.mse_indices", [rows, mse_cols], "uint8"),
        "qjl_signs": (f"{prefix}.qjl_signs", [rows, qjl_cols], "uint8"),
        "norms": (f"{prefix}.norms", [rows], "float32"),
        "residual_norms": (f"{prefix}.residual_norms", [rows], "float32"),
        "pi": (f"{prefix}.pi", [vector_dim, vector_dim], "float32"),
    }
    records: dict[str, dict[str, Any]] = {}
    with safe_open(str(artifact_path), framework="pt", device="cpu") as handle:
        for key, (name, shape, dtype) in specs.items():
            tensor = handle.get_tensor(name).numpy()
            array = np.ascontiguousarray(tensor.astype(np.dtype(dtype), copy=False))
            payload = array.tobytes()
            records[key] = {
                "name": name,
                "shape": shape,
                "dtype": dtype,
                "bytes": len(payload),
                "sha256": sha256_bytes(payload),
            }
    return records


def build_entry_manifest(
    entry: dict[str, Any],
    catalog: dict[str, Any],
    bridge_url: str,
) -> dict[str, Any]:
    """Build a sidecar-compatible manifest for one promoted catalog entry."""
    prefix = entry["prefix"]
    artifact_path = entry["artifact_path"]
    vector_dim = entry["vector_dim"]
    key_bits = entry["key_bits"]
    rows = entry["rows"]
    # tensor shapes follow TurboQuant packing convention
    mse_cols = vector_dim // (8 // key_bits)
    qjl_cols = vector_dim // 8
    tensor_specs = catalog_tensor_records(
        Path(artifact_path),
        prefix,
        rows,
        vector_dim,
        mse_cols,
        qjl_cols,
    )
    stored_tensor_bytes = sum(int(record["bytes"]) for record in tensor_specs.values())

    kernel_path = (PACKAGE_DIR / "turboquant_inverse_q4.opencl").resolve()
    kernel_text = kernel_path.read_text(encoding="utf-8")

    return {
        "schema_version": 1,
        "package": f"catalog_entry_{prefix}",
        "purpose": f"catalog decode verification for {prefix}",
        "artifact": {
            "path": artifact_path,
            "bytes": entry["artifact_bytes"],
            "sha256": entry["artifact_sha256"],
        },
        "prefix": prefix,
        "kernel": {
            "path": str(kernel_path),
            "function": "turboquant_inverse_q4",
            "language": "OpenCL C",
            "bytes": len(kernel_text.encode()),
            "sha256": hashlib.sha256(kernel_text.encode()).hexdigest(),
        },
        "runtime": {
            "bridge_url": bridge_url,
        },
        "vector_dim": vector_dim,
        "key_bits": key_bits,
        "mse_bits": 3,
        "qjl_scale": math.sqrt(math.pi / 2.0) / vector_dim,
        "seeds": {"turboquant_seed": 42, "qjl_seed": 1042},
        "generated_inputs": {
            "centroids": {
                "name": "lloyd_max_centroids",
                "path": str(DEFAULT_NATIVE_CENTROIDS.resolve()),
                "shape": [8],
                "dtype": "float32",
                "bytes": 32,
                "sha256": "67b1b9c205700b78b4bd2683ae7aae8b22498f6c3977c1e00506bc3ecef6aff6",
            },
            "qjl_matrix": {
                "name": "qjl_matrix_s",
                "path": str(DEFAULT_NATIVE_QJL_MATRIX.resolve()),
                "shape": [vector_dim, vector_dim],
                "dtype": "float32",
                "bytes": vector_dim * vector_dim * 4,
                "sha256": "3b24cd980837306d050371917000143d9db8d2bb1c149ccbbaa3886c9ae01c1d",
            },
        },
        "tensors": tensor_specs,
        "loader_conversions": {
            "mse_indices": "uint8 passthrough",
            "qjl_signs": "uint8 passthrough",
            "norms": "cast to float32 before upload",
            "residual_norms": "cast to float32 before upload",
            "pi": "cast to float32 before upload",
        },
        "byte_accounting": {
            "stored_tensor_bytes": stored_tensor_bytes,
            "generated_matrix_bytes": 32 + vector_dim * vector_dim * 4,
            "kernel_bytes": len(kernel_text.encode()),
            "decoded_output_bytes": rows * vector_dim * 4,
        },
        "claim_boundary": (
            "catalog decode verification for a single promoted shardlet"
        ),
    }

def verify_artifact_integrity(entry: dict[str, Any]) -> dict[str, Any]:
    """Hash-check a catalog artifact without decoding."""
    path = Path(entry["artifact_path"])
    result: dict[str, Any] = {
        "prefix": entry["prefix"],
        "artifact_path": str(path),
        "expected_sha256": entry["artifact_sha256"],
        "expected_bytes": entry["artifact_bytes"],
    }
    if not path.exists():
        result["ok"] = False
        result["error"] = "artifact file not found"
        return result
    actual_bytes = path.stat().st_size
    actual_sha256 = sha256_file(path)
    result["actual_bytes"] = actual_bytes
    result["actual_sha256"] = actual_sha256
    result["bytes_match"] = actual_bytes == entry["artifact_bytes"]
    result["sha256_match"] = actual_sha256 == entry["artifact_sha256"]
    result["ok"] = result["bytes_match"] and result["sha256_match"]
    return result


def run_catalog_sidecar(
    entry: dict[str, Any],
    catalog: dict[str, Any],
    bridge_url: str,
    timeout_s: float,
) -> dict[str, Any]:
    """Run sidecar decode for one catalog entry via subprocess."""
    prefix = entry["prefix"]
    run_dir = CATALOG_RESULTS_DIR / prefix.replace(".", "_")
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_entry_manifest(entry, catalog, bridge_url)
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    output_f32 = run_dir / "decoded.f32"
    metadata_path = run_dir / "decode_metadata.json"
    validation_path = run_dir / "runtime_package_validation.json"

    result = run_command(
        f"sidecar decode {prefix}",
        [
            sys.executable,
            str(PACKAGE_DIR / "ggml_opencl_sidecar.py"),
            "--manifest",
            str(manifest_path),
            "--bridge-url",
            bridge_url,
            "--output",
            str(output_f32),
            "--metadata",
            str(metadata_path),
            "--validation-output",
            str(validation_path),
            "--timeout-s",
            str(timeout_s),
        ],
        PACKAGE_DIR,
    )

    entry_result: dict[str, Any] = {
        "prefix": prefix,
        "source_family": entry["source_family"],
        "rows": entry["rows"],
        "backend": "python_sidecar",
        "command": result.to_json(),
        "manifest_path": str(manifest_path),
        "metadata_path": str(metadata_path),
        "validation_path": str(validation_path),
        "output_path": str(output_f32),
    }

    if result.ok and metadata_path.exists():
        metadata = load_json(metadata_path)
        entry_result["decode_ok"] = metadata.get("ok", False)
        entry_result["output_sha256"] = metadata.get("output_sha256", "")
        entry_result["output_bytes"] = metadata.get("output_bytes", 0)
        entry_result["rows_decoded"] = metadata.get("rows", 0)
        entry_result["sample"] = metadata.get("sample", [])
    else:
        entry_result["decode_ok"] = False
        entry_result["error"] = result.stderr_tail or result.stdout_tail

    return entry_result


def compare_catalog_decodes(
    native_decode: dict[str, Any],
    sidecar_decode: dict[str, Any],
) -> dict[str, Any]:
    """Compare native C and Python sidecar decode outputs for one entry."""
    native_sha = native_decode.get("output_sha256", "")
    sidecar_sha = sidecar_decode.get("output_sha256", "")
    native_bytes = native_decode.get("output_file_bytes", native_decode.get("output_bytes", 0))
    sidecar_bytes = sidecar_decode.get("output_bytes", 0)
    native_rows = native_decode.get("rows_decoded", 0)
    sidecar_rows = sidecar_decode.get("rows_decoded", 0)
    ok = (
        bool(native_decode.get("decode_ok"))
        and bool(sidecar_decode.get("decode_ok"))
        and native_sha == sidecar_sha
        and native_bytes == sidecar_bytes
        and native_rows == sidecar_rows
    )
    return {
        "ok": ok,
        "native_output_path": native_decode.get("output_path", ""),
        "native_output_sha256": native_sha,
        "python_sidecar_output_path": sidecar_decode.get("output_path", ""),
        "python_sidecar_output_sha256": sidecar_sha,
        "bytes_match": native_bytes == sidecar_bytes,
        "sha256_match": native_sha == sidecar_sha,
        "rows_match": native_rows == sidecar_rows,
        "native_bytes": native_bytes,
        "python_sidecar_bytes": sidecar_bytes,
        "native_rows": native_rows,
        "python_sidecar_rows": sidecar_rows,
        "sidecar_decode": sidecar_decode,
    }


def run_catalog_native_http(
    entry: dict[str, Any],
    catalog: dict[str, Any],
    bridge_url: str,
    decoded_f32: Path,
) -> dict[str, Any]:
    """Run native C http_bridge decode for one catalog entry."""
    prefix = entry["prefix"]
    run_dir = CATALOG_RESULTS_DIR / prefix.replace(".", "_")
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_entry_manifest(entry, catalog, bridge_url)
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    audit_path = run_dir / "native_http_bridge_audit.json"
    output_path = run_dir / "native_decoded.f32"
    result = run_command(
        f"native http_bridge decode {prefix}",
        [
            "./smoke_ggml_tq_tensor_shim",
            str(decoded_f32),
            str(audit_path),
            "http_bridge",
            str(manifest_path),
            bridge_url,
            str(output_path),
        ],
        PRIVATE_RUNNER_DIR,
    )

    entry_result: dict[str, Any] = {
        "prefix": prefix,
        "source_family": entry["source_family"],
        "rows": entry["rows"],
        "backend": "native_http_bridge",
        "command": result.to_json(),
        "manifest_path": str(manifest_path),
        "audit_path": str(audit_path),
        "output_path": str(output_path),
    }
    if result.ok and audit_path.exists():
        audit = load_json(audit_path)
        tensor = audit.get("tensor", {})
        entry_result["decode_ok"] = bool(audit.get("ok"))
        entry_result["output_bytes"] = tensor.get("nbytes", 0)
        if output_path.exists():
            entry_result["output_file_bytes"] = output_path.stat().st_size
            entry_result["output_sha256"] = sha256_file(output_path)
        entry_result["rows_decoded"] = tensor.get("rows", 0)
        entry_result["cols_decoded"] = tensor.get("cols", 0)
        entry_result["l2_norm"] = tensor.get("l2_norm", 0.0)
        entry_result["sample"] = audit.get("sample", [])
    else:
        entry_result["decode_ok"] = False
        entry_result["error"] = result.stderr_tail or result.stdout_tail
    return entry_result


def run_catalog_mode(args: argparse.Namespace) -> dict[str, Any]:
    """Iterate promoted catalog entries: verify integrity, optionally decode."""
    catalog = load_json(args.catalog)
    promoted = [e for e in catalog["promoted"] if e.get("promoted", False)]

    if not promoted:
        raise RuntimeError("no promoted entries in catalog")

    print(f"catalog mode: {len(promoted)} promoted entries")
    CATALOG_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    bridge_url = args.bridge_url or "http://127.0.0.1:8504"
    bridge_live = False
    if args.probe_live_bridge:
        probe = probe_bridge_status(bridge_url, args.bridge_timeout)
        bridge_live = probe.get("ok", False)
        print(f"  bridge probe: {'live' if bridge_live else 'offline'}")

    integrity_results: list[dict[str, Any]] = []
    decode_results: list[dict[str, Any]] = []
    native_build: dict[str, Any] | str = "not needed"
    all_ok = True

    if bridge_live and args.run_decode and args.catalog_backend == "native_http_bridge":
        build_result = run_command("make native runner", ["make"], PRIVATE_RUNNER_DIR)
        native_build = build_result.to_json()
        if not build_result.ok:
            raise RuntimeError(
                "native runner build failed: "
                f"{build_result.stderr_tail or build_result.stdout_tail}"
            )

    for entry in promoted:
        prefix = entry["prefix"]

        # Gate 1: artifact integrity
        integrity = verify_artifact_integrity(entry)
        integrity_results.append(integrity)
        if not integrity["ok"]:
            print(f"  {prefix}: INTEGRITY FAIL")
            all_ok = False
            continue
        print(f"  {prefix}: integrity OK ({integrity['actual_bytes']}B)")

        # Gate 2: sidecar decode (only if bridge is live)
        if bridge_live and args.run_decode:
            if args.catalog_backend == "native_http_bridge":
                decode = run_catalog_native_http(
                    entry, catalog, bridge_url, args.decoded_f32
                )
                if args.catalog_compare_sidecar:
                    sidecar_decode = run_catalog_sidecar(
                        entry, catalog, bridge_url, args.sidecar_timeout
                    )
                    comparison = compare_catalog_decodes(decode, sidecar_decode)
                    decode["python_sidecar_comparison"] = comparison
            else:
                decode = run_catalog_sidecar(
                    entry, catalog, bridge_url, args.sidecar_timeout
                )
            decode_results.append(decode)
            tag = "DECODE OK" if decode["decode_ok"] else "DECODE FAIL"
            rows = decode.get("rows_decoded", "?")
            print(f"  {prefix}: {tag} ({rows} rows)")
            comparison = decode.get("python_sidecar_comparison")
            if isinstance(comparison, dict):
                compare_tag = "COMPARE OK" if comparison.get("ok") else "COMPARE FAIL"
                print(f"  {prefix}: {compare_tag} (native vs Python sidecar)")
            if not decode["decode_ok"]:
                all_ok = False
            if isinstance(comparison, dict) and not comparison.get("ok"):
                all_ok = False
        elif not bridge_live:
            print(f"  {prefix}: decode skipped (bridge offline)")

    report: dict[str, Any] = {
        "ok": all_ok,
        "catalog_name": catalog.get("catalog_name", ""),
        "promoted_count": len(promoted),
        "integrity_checked": len(integrity_results),
        "integrity_passed": sum(1 for r in integrity_results if r["ok"]),
        "decode_attempted": len(decode_results),
        "decode_passed": sum(1 for r in decode_results if r.get("decode_ok")),
        "comparison_attempted": sum(
            1 for r in decode_results if isinstance(r.get("python_sidecar_comparison"), dict)
        ),
        "comparison_passed": sum(
            1
            for r in decode_results
            if isinstance(r.get("python_sidecar_comparison"), dict)
            and r["python_sidecar_comparison"].get("ok")
        ),
        "catalog_backend": args.catalog_backend,
        "native_build": native_build,
        "bridge_live": bridge_live,
        "bridge_url": bridge_url,
        "integrity": integrity_results,
        "decodes": decode_results,
    }

    report_path = args.catalog_report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument(
        "--sidecar-metadata",
        type=Path,
        default=DEFAULT_SIDECAR_METADATA,
    )
    parser.add_argument("--decoded-f32", type=Path, default=DEFAULT_F32)
    parser.add_argument("--descriptor", type=Path, default=DEFAULT_DESCRIPTOR)
    parser.add_argument(
        "--runner-audit",
        type=Path,
        default=DEFAULT_RUNNER_AUDIT,
    )
    parser.add_argument(
        "--loader-audit",
        type=Path,
        default=DEFAULT_LOADER_AUDIT,
    )
    parser.add_argument(
        "--skip-package-validation",
        dest="validate_package",
        action="store_false",
    )
    parser.add_argument(
        "--no-boundary-check",
        dest="check_backend_boundaries",
        action="store_false",
    )
    parser.add_argument(
        "--skip-live-bridge-probe",
        dest="probe_live_bridge",
        action="store_false",
    )
    parser.add_argument("--require-live-bridge", action="store_true")
    parser.add_argument("--bridge-timeout", type=float, default=5.0)
    parser.add_argument("--stable-audit", action="store_true")

    # catalog mode
    parser.add_argument(
        "--catalog",
        type=Path,
        default=None,
        help="Run in catalog mode: verify and decode all promoted entries",
    )
    parser.add_argument(
        "--catalog-report",
        type=Path,
        default=DEFAULT_CATALOG_REPORT,
    )
    parser.add_argument("--bridge-url", default="")
    parser.add_argument("--sidecar-timeout", type=float, default=60.0)
    parser.add_argument(
        "--catalog-backend",
        choices=("native_http_bridge", "python_sidecar"),
        default="native_http_bridge",
    )
    parser.add_argument(
        "--no-catalog-compare-sidecar",
        dest="catalog_compare_sidecar",
        action="store_false",
        help="Skip native-vs-Python byte comparison for native catalog decodes.",
    )
    parser.add_argument(
        "--no-decode",
        dest="run_decode",
        action="store_false",
    )

    parser.set_defaults(
        validate_package=True,
        check_backend_boundaries=True,
        probe_live_bridge=True,
        run_decode=True,
        catalog_compare_sidecar=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # catalog mode: iterate promoted entries
    if args.catalog is not None:
        report = run_catalog_mode(args)
        passed = report["integrity_passed"]
        total = report["integrity_checked"]
        decoded = report["decode_passed"]
        attempted = report["decode_attempted"]
        compared = report.get("comparison_passed", 0)
        comparison_attempted = report.get("comparison_attempted", 0)
        print(f"\ncatalog run complete")
        print(f"  integrity: {passed}/{total} passed")
        if attempted > 0:
            print(f"  decode: {decoded}/{attempted} passed")
            if comparison_attempted > 0:
                print(f"  compare: {compared}/{comparison_attempted} native-vs-sidecar passed")
        else:
            print(f"  decode: skipped (bridge {'offline' if not report['bridge_live'] else 'not requested'})")
        print(f"  report: {args.catalog_report}")
        if not report["ok"]:
            sys.exit(1)
        return

    # original single-entry mode
    audit = build_framework_audit(args)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    tensor = audit["descriptor"]["tensor"]
    print("python runner framework complete")
    print(f"  audit: {args.audit}")
    print(f"  tensor: {tensor['name']}")
    print(f"  ggml_ne: {tensor['ggml_ne']}")
    print(f"  ggml_nb: {tensor['ggml_nb']}")
    loader = audit["ggml_loader_surface"]
    print(f"  ggml loader surface: {loader['ok']}")
    print("  backend decoded_f32_file: implemented")
    http_bridge = audit["backend_boundaries"]["http_bridge"]
    if isinstance(http_bridge, dict):
        print(f"  backend http_bridge: {http_bridge['mode']}")
    else:
        print("  backend http_bridge: not checked")
    live_bridge = audit["live_bridge"]
    if isinstance(live_bridge, dict):
        print(f"  live bridge: {live_bridge['ok']}")
        print(f"  bridge memory clean: {live_bridge['memory_clean']}")
    print("  rust port: notes captured in framework_notes")


if __name__ == "__main__":
    main()
