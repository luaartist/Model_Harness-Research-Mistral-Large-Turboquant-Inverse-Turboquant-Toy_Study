#!/usr/bin/env python3
"""Decode a validated shardlet package through the AMD OpenCL bridge.

This sidecar is intentionally external to llama.cpp. It consumes the runtime
manifest, validates the package, emits raw float32 rows, and records enough
metadata for a GGML/GGUF loader shim to consume the decoded block later.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


PACKAGE_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = PACKAGE_DIR.parent
RESULTS_DIR = EXPERIMENT_DIR / "results"
MODEL_LAB = Path(__file__).resolve().parents[3]
TURBOQUANT_ROOT = Path(
    os.environ.get("TURBOQUANT_ROOT", str(MODEL_LAB / "external/0xSero_turboquant"))
)
DEFAULT_MANIFEST = PACKAGE_DIR / "runtime_package_manifest.json"
DEFAULT_OUTPUT = RESULTS_DIR / "ggml_opencl_sidecar_decode.f32"
DEFAULT_METADATA = RESULTS_DIR / "ggml_opencl_sidecar_decode.json"

sys.path.insert(0, str(TURBOQUANT_ROOT))
sys.path.insert(0, str(PACKAGE_DIR))
np = importlib.import_module("numpy")
torch = importlib.import_module("torch")
safe_open = importlib.import_module("safetensors").safe_open
validate_module = importlib.import_module("validate_runtime_package")
validate_manifest = validate_module.validate_manifest
resolve_manifest_path = validate_module.resolve_manifest_path


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


class BridgeClient:
    def __init__(self, base_url: str, timeout_s: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def request_json(
        self,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> Any:
        data = None
        method = "GET"
        headers = {}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            method = "POST"
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers=headers,
            method=method,
        )
        with urllib.request.urlopen(
            request,
            timeout=self.timeout_s,
        ) as response:
            return json.loads(response.read().decode("utf-8"))

    def request_bytes(self, path: str, payload: bytes) -> Any:
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=payload,
            method="POST",
        )
        request.add_header("Content-Type", "application/octet-stream")
        with urllib.request.urlopen(
            request,
            timeout=self.timeout_s,
        ) as response:
            return json.loads(response.read().decode("utf-8"))

    def binary_response(self, path: str, payload: dict[str, Any]) -> bytes:
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(
            request,
            timeout=self.timeout_s,
        ) as response:
            return response.read()

    def status(self) -> dict[str, Any]:
        try:
            return self.request_json("/status")
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"bridge is not reachable at {self.base_url}"
            ) from exc

    def malloc(self, size: int) -> int:
        response = self.request_json("/hip/malloc", {"size": int(size)})
        if response.get("result") != 0:
            raise RuntimeError(f"hip/malloc failed: {response}")
        return int(response["handle"])

    def free(self, handle: int) -> None:
        self.request_json("/hip/free", {"handle": int(handle)})

    def upload(self, handle: int, data: bytes) -> None:
        query = urllib.parse.urlencode({"handle": int(handle)})
        response = self.request_bytes(f"/hip/memcpy_htod?{query}", data)
        if response.get("result") != 0:
            raise RuntimeError(f"hip/memcpy_htod failed: {response}")

    def download(self, handle: int, size: int) -> bytes:
        return self.binary_response(
            "/hip/memcpy_dtoh",
            {"handle": int(handle), "size": int(size)},
        )

    def load_function(self, source: str, name: str) -> int:
        module_response = self.request_json(
            "/hip/module_load",
            {"source": source},
        )
        if module_response.get("result") != 0:
            raise RuntimeError(f"module_load failed: {module_response}")
        function_response = self.request_json(
            "/hip/module_get_function",
            {"module": int(module_response["module"]), "name": name},
        )
        if function_response.get("result") != 0:
            raise RuntimeError(
                f"module_get_function failed: {function_response}"
            )
        return int(function_response["function"])


def load_package_tensors(
    manifest: dict[str, Any],
    row_limit: int | None,
) -> dict[str, Any]:
    artifact_path = resolve_manifest_path(manifest["artifact"]["path"])
    arrays: dict[str, Any] = {}
    with safe_open(str(artifact_path), framework="pt", device="cpu") as handle:
        for key, record in manifest["tensors"].items():
            tensor = handle.get_tensor(record["name"]).numpy()
            dtype = np.dtype(record["dtype"])
            array = np.ascontiguousarray(tensor.astype(dtype, copy=False))
            if row_limit is not None and key != "pi" and array.ndim >= 1:
                array = np.ascontiguousarray(array[:row_limit])
            arrays[key] = array
    return arrays


def generated_inputs(manifest: dict[str, Any]) -> tuple[Any, Any]:
    centroid_record = manifest["generated_inputs"]["centroids"]
    qjl_record = manifest["generated_inputs"]["qjl_matrix"]
    centroids = np.fromfile(
        resolve_manifest_path(centroid_record["path"]),
        dtype=np.dtype(centroid_record["dtype"]),
    ).reshape(centroid_record["shape"])
    qjl_matrix = np.fromfile(
        resolve_manifest_path(qjl_record["path"]),
        dtype=np.dtype(qjl_record["dtype"]),
    ).reshape(qjl_record["shape"])
    return np.ascontiguousarray(centroids), np.ascontiguousarray(qjl_matrix)


def upload_array(client: BridgeClient, handles: list[int], array: Any) -> int:
    handle = client.malloc(int(array.nbytes))
    handles.append(handle)
    client.upload(handle, array.tobytes())
    return handle


def launch_decode(
    client: BridgeClient,
    function: int,
    manifest: dict[str, Any],
    tensors: dict[str, Any],
    centroids: Any,
    qjl_matrix: Any,
) -> Any:
    rows = int(tensors["mse_indices"].shape[0])
    vector_dim = int(manifest["vector_dim"])
    output = np.empty((rows, vector_dim), dtype=np.float32)
    mse_cols = int(tensors["mse_indices"].shape[1])
    qjl_cols = int(tensors["qjl_signs"].shape[1])
    qjl_scale = float(
        manifest.get("qjl_scale") or math.sqrt(math.pi / 2.0) / vector_dim
    )
    handles: list[int] = []
    try:
        mse_handle = upload_array(client, handles, tensors["mse_indices"])
        qjl_handle = upload_array(client, handles, tensors["qjl_signs"])
        norm_handle = upload_array(client, handles, tensors["norms"])
        residual_handle = upload_array(
            client,
            handles,
            tensors["residual_norms"],
        )
        centroid_handle = upload_array(client, handles, centroids)
        pi_handle = upload_array(client, handles, tensors["pi"])
        qjl_matrix_handle = upload_array(client, handles, qjl_matrix)
        output_handle = client.malloc(int(output.nbytes))
        handles.append(output_handle)

        block = 128
        grid = math.ceil((rows * vector_dim) / block)
        response = client.request_json(
            "/hip/launch_kernel",
            {
                "function": int(function),
                "grid": [int(grid)],
                "block": [int(block)],
                "args": [
                    {"kind": "buffer", "handle": int(mse_handle)},
                    {"kind": "buffer", "handle": int(qjl_handle)},
                    {"kind": "buffer", "handle": int(norm_handle)},
                    {"kind": "buffer", "handle": int(residual_handle)},
                    {"kind": "buffer", "handle": int(centroid_handle)},
                    {"kind": "buffer", "handle": int(pi_handle)},
                    {"kind": "buffer", "handle": int(qjl_matrix_handle)},
                    {"kind": "buffer", "handle": int(output_handle)},
                    {"kind": "i32", "value": int(rows)},
                    {"kind": "i32", "value": int(mse_cols)},
                    {"kind": "i32", "value": int(qjl_cols)},
                    {"kind": "i32", "value": int(vector_dim)},
                    {"kind": "f32", "value": qjl_scale},
                ],
            },
        )
        if response.get("result") != 0:
            raise RuntimeError(f"launch_kernel failed: {response}")
        output_bytes = client.download(output_handle, int(output.nbytes))
        return np.frombuffer(
            output_bytes,
            dtype=np.float32,
        ).reshape(output.shape)
    finally:
        for handle in reversed(handles):
            client.free(handle)


def run(args: argparse.Namespace) -> dict[str, Any]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    validation = validate_manifest(args.manifest, args.validation_output)
    if not validation["valid"]:
        raise RuntimeError(
            f"manifest validation failed: {validation['failures']}"
        )

    manifest = load_manifest(args.manifest)
    bridge_url = args.bridge_url or manifest["runtime"]["bridge_url"]
    client = BridgeClient(bridge_url, args.timeout_s)
    bridge_status = client.status()

    rows_arg = args.rows if args.rows and args.rows > 0 else None
    tensors = load_package_tensors(manifest, rows_arg)
    centroids, qjl_matrix = generated_inputs(manifest)
    kernel_source = resolve_manifest_path(manifest["kernel"]["path"]).read_text(
        encoding="utf-8"
    )
    function = client.load_function(
        kernel_source,
        manifest["kernel"]["function"],
    )
    decoded = launch_decode(
        client,
        function,
        manifest,
        tensors,
        centroids,
        qjl_matrix,
    )

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_bytes = np.ascontiguousarray(decoded).tobytes()
    output_path.write_bytes(output_bytes)
    metadata = {
        "ok": True,
        "adapter": "ggml-opencl-sidecar",
        "manifest": str(args.manifest),
        "package": manifest["package"],
        "prefix": manifest["prefix"],
        "bridge_url": bridge_url,
        "bridge_status": bridge_status,
        "artifact": manifest["artifact"],
        "kernel": manifest["kernel"],
        "rows": int(decoded.shape[0]),
        "vector_dim": int(decoded.shape[1]),
        "dtype": "float32",
        "layout": "row-major contiguous f32",
        "output": str(output_path),
        "output_bytes": len(output_bytes),
        "output_sha256": sha256_bytes(output_bytes),
        "sample": decoded.reshape(-1)[:8].tolist(),
        "claim_boundary": manifest.get("claim_boundary", ""),
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--bridge-url", default="")
    parser.add_argument("--rows", type=int, default=0)
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument(
        "--validation-output",
        type=Path,
        default=RESULTS_DIR / "runtime_package_validation.json",
    )
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    print("ggml-opencl sidecar decode complete")
    print(f"  package: {result['package']}")
    print(f"  prefix: {result['prefix']}")
    print(f"  rows: {result['rows']}")
    print(f"  output: {result['output']}")
    print(f"  output_bytes: {result['output_bytes']}")
    print(f"  output_sha256: {result['output_sha256']}")


if __name__ == "__main__":
    main()
