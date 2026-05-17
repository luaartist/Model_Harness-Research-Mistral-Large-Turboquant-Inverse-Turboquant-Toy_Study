#!/usr/bin/env python3
"""Run a bridge-side decode primitive over a real packed shardlet."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import safe_open


TOOL_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = TOOL_DIR.parent
RESULTS_DIR = EXPERIMENT_DIR / "results"
MODEL_LAB = Path(os.environ.get("MODEL_LAB", "../Mistral-Large-Quantization-Project/_model_lab"))
DEFAULT_ARTIFACT = (
    MODEL_LAB
    / "experiments/0018_same_harness_prior_shootout/artifacts/"
    "prior_shootout_q4_shardlet.safetensors"
)
TURBOQUANT_ROOT = MODEL_LAB / "external/0xSero_turboquant"
BRIDGE_URL = os.environ.get("BRIDGE_URL", "http://127.0.0.1:8504")
VECTOR_DIM = 128

sys.path.insert(0, str(TURBOQUANT_ROOT))


def load_codebook_helper() -> object:
    from turboquant.codebook import get_codebook

    return get_codebook


KERNEL_SOURCE = r"""
__kernel void decode_mse_q4_proxy(
    __global const uchar * packed,
    __global const float * norms,
    __global const float * centroids,
    __global float * output,
    const int rows,
    const int packed_cols,
    const int vector_dim
) {
    int gid = get_global_id(0);
    int total = rows * vector_dim;
    if (gid >= total) {
        return;
    }
    int row = gid / vector_dim;
    int col = gid - row * vector_dim;
    uchar byte_val = packed[row * packed_cols + (col >> 1)];
    uchar code = (col & 1) == 0 ? (byte_val & 15) : ((byte_val >> 4) & 15);
    output[gid] = centroids[(int) code] * norms[row];
}
"""


def request_json(path: str, payload: dict[str, Any] | None = None) -> Any:
    data = None
    method = "GET"
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        method = "POST"
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        f"{BRIDGE_URL}{path}",
        data=data,
        headers=headers,
        method=method,
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def request_bytes(path: str, payload: bytes) -> Any:
    request = urllib.request.Request(
        f"{BRIDGE_URL}{path}",
        data=payload,
        method="POST",
    )
    request.add_header("Content-Type", "application/octet-stream")
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def post_binary_response(path: str, payload: dict[str, Any]) -> bytes:
    request = urllib.request.Request(
        f"{BRIDGE_URL}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def bridge_alive() -> dict[str, Any]:
    try:
        return request_json("/status")
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "HIP bridge is not reachable at http://127.0.0.1:8504"
        ) from exc


def malloc(size: int) -> int:
    response = request_json("/hip/malloc", {"size": int(size)})
    if response.get("result") != 0:
        raise RuntimeError(f"hip/malloc failed: {response}")
    return int(response["handle"])


def free(handle: int) -> None:
    request_json("/hip/free", {"handle": int(handle)})


def upload(handle: int, data: bytes) -> None:
    query = urllib.parse.urlencode({"handle": int(handle)})
    response = request_bytes(f"/hip/memcpy_htod?{query}", data)
    if response.get("result") != 0:
        raise RuntimeError(f"hip/memcpy_htod failed: {response}")


def download(handle: int, size: int) -> bytes:
    return post_binary_response(
        "/hip/memcpy_dtoh",
        {"handle": int(handle), "size": int(size)},
    )


def load_function(source: str, name: str) -> int:
    module_response = request_json("/hip/module_load", {"source": source})
    if module_response.get("result") != 0:
        raise RuntimeError(f"module_load failed: {module_response}")
    function_response = request_json(
        "/hip/module_get_function",
        {"module": int(module_response["module"]), "name": name},
    )
    if function_response.get("result") != 0:
        raise RuntimeError(f"module_get_function failed: {function_response}")
    return int(function_response["function"])


def launch(
    function: int,
    rows: int,
    packed_cols: int,
    vector_dim: int,
    packed_handle: int,
    norms_handle: int,
    centroids_handle: int,
    output_handle: int,
) -> None:
    total = rows * vector_dim
    block = 256
    grid = math.ceil(total / block)
    payload = {
        "function": int(function),
        "grid": [int(grid)],
        "block": [int(block)],
        "args": [
            {"kind": "buffer", "handle": int(packed_handle)},
            {"kind": "buffer", "handle": int(norms_handle)},
            {"kind": "buffer", "handle": int(centroids_handle)},
            {"kind": "buffer", "handle": int(output_handle)},
            {"kind": "i32", "value": int(rows)},
            {"kind": "i32", "value": int(packed_cols)},
            {"kind": "i32", "value": int(vector_dim)},
        ],
    }
    response = request_json("/hip/launch_kernel", payload)
    if response.get("result") != 0:
        raise RuntimeError(f"launch_kernel failed: {response}")


def choose_prefix(path: Path, requested: str | None) -> str:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        prefixes = sorted(
            set(".".join(key.split(".")[:3]) for key in handle.keys())
        )
    if requested:
        if requested not in prefixes:
            raise ValueError(f"prefix {requested} not found in {path}")
        return requested
    for prefix in prefixes:
        if ".q4." in prefix:
            return prefix
    raise ValueError(f"no q4 prefix found in {path}")


def load_shardlet_inputs(
    path: Path,
    prefix: str,
) -> tuple[np.ndarray, np.ndarray]:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        packed = handle.get_tensor(f"{prefix}.mse_indices").numpy()
        norms = handle.get_tensor(f"{prefix}.norms").numpy().astype(np.float32)
    if packed.dtype != np.uint8:
        packed = packed.astype(np.uint8)
    return np.ascontiguousarray(packed), np.ascontiguousarray(norms)


def cpu_decode_proxy(
    packed: np.ndarray,
    norms: np.ndarray,
    centroids: np.ndarray,
    vector_dim: int,
) -> np.ndarray:
    rows = packed.shape[0]
    output = np.empty((rows, vector_dim), dtype=np.float32)
    for row in range(rows):
        for col in range(vector_dim):
            byte_val = int(packed[row, col // 2])
            code = byte_val & 15 if col % 2 == 0 else (byte_val >> 4) & 15
            output[row, col] = centroids[code] * norms[row]
    return output


def run(args: argparse.Namespace) -> dict[str, Any]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    status = bridge_alive()
    prefix = choose_prefix(args.artifact, args.prefix)
    packed, norms = load_shardlet_inputs(args.artifact, prefix)
    rows, packed_cols = packed.shape
    mse_bits = args.key_bits - 1
    get_codebook = load_codebook_helper()
    codebook = get_codebook(VECTOR_DIM, mse_bits)
    centroids = np.asarray(codebook["centroids"], dtype=np.float32)
    cpu_output = cpu_decode_proxy(packed, norms, centroids, VECTOR_DIM)

    handles = []
    try:
        packed_handle = malloc(packed.nbytes)
        norms_handle = malloc(norms.nbytes)
        centroids_handle = malloc(centroids.nbytes)
        output_handle = malloc(cpu_output.nbytes)
        handles.extend(
            [packed_handle, norms_handle, centroids_handle, output_handle]
        )
        upload(packed_handle, packed.tobytes())
        upload(norms_handle, norms.tobytes())
        upload(centroids_handle, centroids.tobytes())
        function = load_function(KERNEL_SOURCE, "decode_mse_q4_proxy")
        launch(
            function,
            rows,
            packed_cols,
            VECTOR_DIM,
            packed_handle,
            norms_handle,
            centroids_handle,
            output_handle,
        )
        gpu_bytes = download(output_handle, cpu_output.nbytes)
    finally:
        for handle in reversed(handles):
            free(handle)

    gpu_output = np.frombuffer(
        gpu_bytes,
        dtype=np.float32,
    ).reshape(cpu_output.shape)
    diff = gpu_output - cpu_output
    result = {
        "experiment": "0021_bridge_shardlet_decode_probe",
        "bridge_status": status,
        "artifact": str(args.artifact),
        "prefix": prefix,
        "rows": int(rows),
        "packed_cols": int(packed_cols),
        "vector_dim": VECTOR_DIM,
        "key_bits": int(args.key_bits),
        "mse_bits": int(mse_bits),
        "packed_bytes": int(packed.nbytes),
        "norm_bytes": int(norms.nbytes),
        "centroid_bytes": int(centroids.nbytes),
        "output_bytes": int(cpu_output.nbytes),
        "max_abs_diff": float(np.max(np.abs(diff))),
        "mean_abs_diff": float(np.mean(np.abs(diff))),
        "allclose": bool(np.allclose(gpu_output, cpu_output, atol=1e-7)),
        "output_sample": gpu_output.reshape(-1)[:8].tolist(),
    }
    output_path = RESULTS_DIR / "bridge_shardlet_decode_probe.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--key-bits", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    print("bridge shardlet decode probe complete")
    print(f"  prefix: {result['prefix']}")
    print(f"  rows: {result['rows']}")
    print(f"  packed bytes: {result['packed_bytes']}")
    print(f"  output bytes: {result['output_bytes']}")
    print(f"  max_abs_diff: {result['max_abs_diff']:.9g}")
    print(f"  allclose: {result['allclose']}")


if __name__ == "__main__":
    main()
