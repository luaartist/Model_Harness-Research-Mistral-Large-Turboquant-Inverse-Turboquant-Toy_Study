#!/usr/bin/env python3
"""Run the full TurboQuant inverse through the HIP/OpenCL bridge."""

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

TOOL_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = TOOL_DIR.parent
RESULTS_DIR = EXPERIMENT_DIR / "results"
PACKAGE_DIR = EXPERIMENT_DIR / "runtime_package"
MODEL_LAB = Path(os.environ.get("MODEL_LAB", "../Mistral-Large-Quantization-Project/_model_lab"))
TURBOQUANT_ROOT = MODEL_LAB / "external/0xSero_turboquant"
DEFAULT_ARTIFACT = (
    MODEL_LAB
    / "experiments/0018_same_harness_prior_shootout/artifacts/"
    "prior_shootout_q4_shardlet.safetensors"
)
DEFAULT_PREFIX = "c0.q4.flavor_best"
BRIDGE_URL = os.environ.get("BRIDGE_URL", "http://127.0.0.1:8504")
VECTOR_DIM = 128

sys.path.insert(0, str(TURBOQUANT_ROOT))
np = importlib.import_module("numpy")
torch = importlib.import_module("torch")
safe_open = importlib.import_module("safetensors").safe_open


def load_turboquant_helpers() -> dict[str, Any]:
    get_codebook = importlib.import_module("turboquant.codebook").get_codebook
    quantizer_module = importlib.import_module("turboquant.quantizer")
    generate_qjl_matrix = importlib.import_module(
        "turboquant.rotation"
    ).generate_qjl_matrix
    return {
        "get_codebook": get_codebook,
        "ProdQuantized": quantizer_module.ProdQuantized,
        "TurboQuantProd": quantizer_module.TurboQuantProd,
        "generate_qjl_matrix": generate_qjl_matrix,
    }


KERNEL_SOURCE = r"""
__kernel void turboquant_inverse_q4(
    __global const uchar * mse_packed,
    __global const uchar * qjl_packed,
    __global const float * norms,
    __global const float * residual_norms,
    __global const float * centroids,
    __global const float * pi,
    __global const float * s_matrix,
    __global float * output,
    const int rows,
    const int mse_packed_cols,
    const int qjl_packed_cols,
    const int vector_dim,
    const float qjl_scale
) {
    int gid = get_global_id(0);
    int total = rows * vector_dim;
    if (gid >= total) {
        return;
    }

    int row = gid / vector_dim;
    int col = gid - row * vector_dim;

    float mse_acc = 0.0f;
    float qjl_acc = 0.0f;
    for (int j = 0; j < vector_dim; ++j) {
        uchar packed_byte = mse_packed[row * mse_packed_cols + (j >> 1)];
        uchar code = (j & 1) == 0 ?
            (packed_byte & 15) : ((packed_byte >> 4) & 15);
        float y = centroids[(int) code];
        mse_acc += y * pi[j * vector_dim + col];

        uchar sign_byte = qjl_packed[row * qjl_packed_cols + (j >> 3)];
        float sign = ((sign_byte >> (j & 7)) & 1) ? 1.0f : -1.0f;
        qjl_acc += sign * s_matrix[j * vector_dim + col];
    }

    float mse_part = mse_acc * norms[row];
    float qjl_part = qjl_acc * qjl_scale * residual_norms[row];
    output[gid] = mse_part + qjl_part;
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


def load_prefix_tensors(path: Path, prefix: str) -> dict[str, np.ndarray]:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        tensors = {
            "mse_indices": handle.get_tensor(
                f"{prefix}.mse_indices"
            ).numpy(),
            "qjl_signs": handle.get_tensor(f"{prefix}.qjl_signs").numpy(),
            "norms": handle.get_tensor(f"{prefix}.norms").numpy(),
            "residual_norms": handle.get_tensor(
                f"{prefix}.residual_norms"
            ).numpy(),
            "pi": handle.get_tensor(f"{prefix}.pi").numpy(),
        }
    return {
        "mse_indices": np.ascontiguousarray(
            tensors["mse_indices"].astype(np.uint8, copy=False)
        ),
        "qjl_signs": np.ascontiguousarray(
            tensors["qjl_signs"].astype(np.uint8, copy=False)
        ),
        "norms": np.ascontiguousarray(tensors["norms"].astype(np.float32)),
        "residual_norms": np.ascontiguousarray(
            tensors["residual_norms"].astype(np.float32)
        ),
        "pi": np.ascontiguousarray(tensors["pi"].astype(np.float32)),
    }


def torch_reference(
    tensors: dict[str, np.ndarray],
    key_bits: int,
    seed: int,
) -> np.ndarray:
    if not torch.cuda.is_available():
        raise RuntimeError("ROCm/CUDA torch is required for reference decode")
    helpers = load_turboquant_helpers()
    TurboQuantProd = helpers["TurboQuantProd"]
    ProdQuantized = helpers["ProdQuantized"]
    device = torch.device("cuda")
    quantizer = TurboQuantProd(
        dim=VECTOR_DIM,
        bits=key_bits,
        device=device,
        dtype=torch.float32,
        seed=seed,
    )
    with torch.no_grad():
        quantizer.mse_quantizer.Pi.copy_(
            torch.from_numpy(tensors["pi"]).to(device=device)
        )
        q_value = ProdQuantized(
            mse_indices=torch.from_numpy(tensors["mse_indices"]).to(device),
            qjl_signs=torch.from_numpy(tensors["qjl_signs"]).to(device),
            residual_norms=torch.from_numpy(
                tensors["residual_norms"]
            ).to(device),
            norms=torch.from_numpy(tensors["norms"]).to(device),
            mse_bits=key_bits - 1,
        )
        reference = quantizer.dequantize(q_value).detach().cpu().numpy()
    return np.ascontiguousarray(reference.astype(np.float32))


def matrix_inputs(key_bits: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    helpers = load_turboquant_helpers()
    get_codebook = helpers["get_codebook"]
    generate_qjl_matrix = helpers["generate_qjl_matrix"]
    codebook = get_codebook(VECTOR_DIM, key_bits - 1)
    centroids = np.asarray(codebook["centroids"], dtype=np.float32)
    s_matrix = generate_qjl_matrix(
        VECTOR_DIM,
        torch.device("cpu"),
        torch.float32,
        seed=seed + 1000,
    ).numpy()
    return np.ascontiguousarray(centroids), np.ascontiguousarray(s_matrix)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_record(name: str, array: np.ndarray) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(array)
    return {
        "name": name,
        "shape": [int(value) for value in contiguous.shape],
        "dtype": str(contiguous.dtype),
        "bytes": int(contiguous.nbytes),
        "sha256": sha256_bytes(contiguous.tobytes()),
    }


def upload_array(handles: list[int], array: np.ndarray) -> int:
    handle = malloc(array.nbytes)
    handles.append(handle)
    upload(handle, array.tobytes())
    return handle


def launch_inverse(
    function: int,
    tensors: dict[str, np.ndarray],
    centroids: np.ndarray,
    s_matrix: np.ndarray,
    output_shape: tuple[int, int],
) -> np.ndarray:
    rows, vector_dim = output_shape
    mse_cols = tensors["mse_indices"].shape[1]
    qjl_cols = tensors["qjl_signs"].shape[1]
    output = np.empty(output_shape, dtype=np.float32)
    qjl_scale = math.sqrt(math.pi / 2.0) / vector_dim
    handles: list[int] = []
    try:
        mse_handle = upload_array(handles, tensors["mse_indices"])
        qjl_handle = upload_array(handles, tensors["qjl_signs"])
        norm_handle = upload_array(handles, tensors["norms"])
        residual_handle = upload_array(handles, tensors["residual_norms"])
        centroid_handle = upload_array(handles, centroids)
        pi_handle = upload_array(handles, tensors["pi"])
        s_handle = upload_array(handles, s_matrix)
        output_handle = malloc(output.nbytes)
        handles.append(output_handle)

        block = 128
        grid = math.ceil((rows * vector_dim) / block)
        response = request_json(
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
                    {"kind": "buffer", "handle": int(s_handle)},
                    {"kind": "buffer", "handle": int(output_handle)},
                    {"kind": "i32", "value": int(rows)},
                    {"kind": "i32", "value": int(mse_cols)},
                    {"kind": "i32", "value": int(qjl_cols)},
                    {"kind": "i32", "value": int(vector_dim)},
                    {"kind": "f32", "value": float(qjl_scale)},
                ],
            },
        )
        if response.get("result") != 0:
            raise RuntimeError(f"launch_kernel failed: {response}")
        gpu_bytes = download(output_handle, output.nbytes)
        return np.frombuffer(gpu_bytes, dtype=np.float32).reshape(output_shape)
    finally:
        for handle in reversed(handles):
            free(handle)


def write_package_manifest(
    result: dict[str, Any],
    tensors: dict[str, np.ndarray],
    centroids: np.ndarray,
    s_matrix: np.ndarray,
) -> None:
    PACKAGE_DIR.mkdir(parents=True, exist_ok=True)
    kernel_path = PACKAGE_DIR / "turboquant_inverse_q4.opencl"
    manifest_path = PACKAGE_DIR / "runtime_package_manifest.json"
    kernel_path.write_text(KERNEL_SOURCE, encoding="utf-8")
    artifact_path = Path(result["artifact"])
    tensor_records = {
        "mse_indices": array_record(
            f"{result['prefix']}.mse_indices",
            tensors["mse_indices"],
        ),
        "qjl_signs": array_record(
            f"{result['prefix']}.qjl_signs",
            tensors["qjl_signs"],
        ),
        "norms": array_record(f"{result['prefix']}.norms", tensors["norms"]),
        "residual_norms": array_record(
            f"{result['prefix']}.residual_norms",
            tensors["residual_norms"],
        ),
        "pi": array_record(f"{result['prefix']}.pi", tensors["pi"]),
    }
    generated_records = {
        "centroids": array_record("lloyd_max_centroids", centroids),
        "qjl_matrix": array_record("qjl_matrix_s", s_matrix),
    }
    stored_tensor_bytes = sum(
        int(record["bytes"]) for record in tensor_records.values()
    )
    generated_matrix_bytes = sum(
        int(record["bytes"]) for record in generated_records.values()
    )
    manifest = {
        "schema_version": 1,
        "package": "mistral_large_tq_runtime_bridge_v0",
        "purpose": "validated q4 TurboQuant shardlet runtime package",
        "created_by_experiment": result["experiment"],
        "artifact": {
            "path": str(artifact_path),
            "bytes": int(artifact_path.stat().st_size),
            "sha256": sha256_file(artifact_path),
        },
        "prefix": result["prefix"],
        "kernel": {
            "path": str(kernel_path),
            "function": "turboquant_inverse_q4",
            "language": "OpenCL C",
            "bytes": int(kernel_path.stat().st_size),
            "sha256": sha256_file(kernel_path),
        },
        "runtime": {
            "bridge_url": BRIDGE_URL,
            "backend": result["bridge_status"].get("backend"),
            "device": result["bridge_status"].get("device"),
            "compute_units": result["bridge_status"].get("compute_units"),
            "vram_total_gb": result["bridge_status"].get("vram_total_gb"),
        },
        "vector_dim": result["vector_dim"],
        "key_bits": result["key_bits"],
        "mse_bits": result["mse_bits"],
        "qjl_scale": result["qjl_scale"],
        "seeds": {
            "turboquant_seed": result["seed"],
            "qjl_seed": result["qjl_seed"],
        },
        "tensors": tensor_records,
        "loader_conversions": {
            "mse_indices": "uint8 passthrough",
            "qjl_signs": "uint8 passthrough",
            "norms": "cast to float32 before upload",
            "residual_norms": "cast to float32 before upload",
            "pi": "cast to float32 before upload",
        },
        "generated_inputs": generated_records,
        "byte_accounting": {
            "stored_tensor_bytes": int(stored_tensor_bytes),
            "generated_matrix_bytes": int(generated_matrix_bytes),
            "kernel_bytes": int(kernel_path.stat().st_size),
            "decoded_output_bytes": result["output_bytes"],
        },
        "validation": {
            "result_path": str(
                RESULTS_DIR / "bridge_turboquant_inverse_probe.json"
            ),
            "max_abs_diff": result["max_abs_diff"],
            "mean_abs_diff": result["mean_abs_diff"],
            "rms_diff": result["rms_diff"],
            "allclose_1e_4": result["allclose_1e_4"],
            "allclose_1e_5": result["allclose_1e_5"],
        },
        "runner_contract": {
            "load": [
                "Open safetensors artifact",
                "Select tensors by prefix",
                "Generate centroids from vector_dim and mse_bits",
                "Generate qjl_matrix from qjl_seed",
                "Compile kernel function turboquant_inverse_q4",
            ],
            "inputs": [
                "mse_indices uint8 [rows, vector_dim / 2]",
                "qjl_signs uint8 [rows, vector_dim / 8]",
                "norms float32 [rows]",
                "residual_norms float32 [rows]",
                "pi float32 [vector_dim, vector_dim]",
                "centroids float32 [2 ** mse_bits]",
                "qjl_matrix float32 [vector_dim, vector_dim]",
            ],
            "output": "float32 [rows, vector_dim] decoded vectors",
            "status": "validated shardlet package, not full-model inference",
        },
        "next_runtime_target": "ggml-opencl sidecar backend",
        "claim_boundary": (
            "This proves package decode correctness for one real q4 shardlet. "
            "It does not yet prove full Mistral inference, native GGML "
            "loading, "
            "or compression superiority over market baselines."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    status = bridge_alive()
    tensors = load_prefix_tensors(args.artifact, args.prefix)
    reference = torch_reference(tensors, args.key_bits, args.seed)
    centroids, s_matrix = matrix_inputs(args.key_bits, args.seed)
    function = load_function(KERNEL_SOURCE, "turboquant_inverse_q4")
    gpu_output = launch_inverse(
        function,
        tensors,
        centroids,
        s_matrix,
        reference.shape,
    )
    diff = gpu_output - reference
    qjl_scale = math.sqrt(math.pi / 2.0) / VECTOR_DIM
    result = {
        "experiment": "0021_bridge_turboquant_inverse_probe",
        "bridge_status": status,
        "artifact": str(args.artifact),
        "prefix": args.prefix,
        "rows": int(reference.shape[0]),
        "vector_dim": int(reference.shape[1]),
        "key_bits": int(args.key_bits),
        "mse_bits": int(args.key_bits - 1),
        "qjl_scale": float(qjl_scale),
        "seed": int(args.seed),
        "qjl_seed": int(args.seed + 1000),
        "mse_packed_bytes": int(tensors["mse_indices"].nbytes),
        "qjl_packed_bytes": int(tensors["qjl_signs"].nbytes),
        "norm_bytes": int(tensors["norms"].nbytes),
        "residual_norm_bytes": int(tensors["residual_norms"].nbytes),
        "pi_bytes": int(tensors["pi"].nbytes),
        "output_bytes": int(reference.nbytes),
        "max_abs_diff": float(np.max(np.abs(diff))),
        "mean_abs_diff": float(np.mean(np.abs(diff))),
        "rms_diff": float(np.sqrt(np.mean(diff * diff))),
        "allclose_1e_4": bool(np.allclose(gpu_output, reference, atol=1e-4)),
        "allclose_1e_5": bool(np.allclose(gpu_output, reference, atol=1e-5)),
        "gpu_output_sample": gpu_output.reshape(-1)[:8].tolist(),
        "torch_reference_sample": reference.reshape(-1)[:8].tolist(),
    }
    output_path = RESULTS_DIR / "bridge_turboquant_inverse_probe.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_package_manifest(result, tensors, centroids, s_matrix)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--key-bits", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    print("bridge TurboQuant inverse probe complete")
    print(f"  prefix: {result['prefix']}")
    print(f"  rows: {result['rows']}")
    print(f"  output bytes: {result['output_bytes']}")
    print(f"  max_abs_diff: {result['max_abs_diff']:.9g}")
    print(f"  rms_diff: {result['rms_diff']:.9g}")
    print(f"  allclose_1e_4: {result['allclose_1e_4']}")


if __name__ == "__main__":
    main()
