#!/usr/bin/env python3
"""Probe the bare-metal bridge + llama.cpp coupling surface."""

from __future__ import annotations

import hashlib
import json
import urllib.request
import os
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXPERIMENT_DIR / "results"

BRIDGE_ROOT = Path(os.environ.get("BRIDGE_ROOT", "../bridge_units"))
LLAMA_ROOT = Path(os.environ.get("LLAMA_ROOT", "third_party/llama.cpp"))
MODEL_LAB = Path(os.environ.get("MODEL_LAB", "../Mistral-Large-Quantization-Project/_model_lab"))

KEY_FILES = {
    "hip_bridge_server": BRIDGE_ROOT
    / "unit12_hip_bridge/src/tph_hip_bridge_server.py",
    "websocket_gpu_bridge": BRIDGE_ROOT
    / "unit12_hip_bridge/src/websocket_gpu_bridge.py",
    "gpu_interceptor": BRIDGE_ROOT
    / "unit12_hip_bridge/src/gpu_interceptor.py",
    "amdhip64_shim": BRIDGE_ROOT / "unit12_hip_bridge/src/amdhip64_shim.c",
    "hip_version_notes": BRIDGE_ROOT
    / "unit12_hip_bridge/src/ZLUDA_SYMBOL_COMPAT_NOTES.md",
    "fake_hipcc": BRIDGE_ROOT / "unit12_hip_bridge/src/fake_hipcc",
    "zluda_cuda": BRIDGE_ROOT / "unit13_zluda/zluda/libcuda.so.1",
    "quantum_bridge": BRIDGE_ROOT
    / "unit14_quantum_bridge/cygwin_quantum_bridge.py",
    "llama_quant": LLAMA_ROOT / "src/llama-quant.cpp",
    "llama_kv_cache": LLAMA_ROOT / "src/llama-kv-cache.cpp",
    "ggml_quants": LLAMA_ROOT / "ggml/src/ggml-quants.c",
    "ggml_opencl": LLAMA_ROOT / "ggml/src/ggml-opencl/ggml-opencl.cpp",
    "ggml_vulkan": LLAMA_ROOT / "ggml/src/ggml-vulkan/ggml-vulkan.cpp",
    "ggml_hip_cmake": LLAMA_ROOT / "ggml/src/ggml-hip/CMakeLists.txt",
    "exp0020_result": MODEL_LAB
    / "experiments/0020_streaming_staggered_prior_run/results/streaming_staggered_prior_result.json",
}

PORTS = {
    "vulkan_tph": (8501, "/status"),
    "opencl_tph": (8502, "/status"),
    "rocm_tph": (8503, "/status"),
    "hip_bridge": (8504, "/status"),
    "gpu_keepalive": (8505, "/gpu/detect"),
}


def sha256_short(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def file_record(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else None,
        "sha256_16": sha256_short(path),
    }


def probe_port(port: int, path: str) -> dict[str, object]:
    url = f"http://127.0.0.1:{port}{path}"
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            payload = response.read(512).decode(errors="replace")
            return {
                "alive": True,
                "url": url,
                "status": response.status,
                "sample": payload[:300],
            }
    except Exception as exc:
        return {
            "alive": False,
            "url": url,
            "error": type(exc).__name__,
            "message": str(exc)[:200],
        }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "experiment": "0021_baremetal_llamacpp_coupling",
        "bridge_root": str(BRIDGE_ROOT),
        "llama_root": str(LLAMA_ROOT),
        "files": {name: file_record(path) for name, path in KEY_FILES.items()},
        "ports": {
            name: probe_port(port, path)
            for name, (port, path) in PORTS.items()
        },
        "coupling_hypothesis": {
            "phase_0": "sidecar OpenCL/HIP bridge quant-dequant kernel probe",
            "phase_1": "llama.cpp quantizer artifact emission hook",
            "phase_2": "KV-cache compression path using llama-kv-cache types",
            "phase_3": "backend kernels in ggml-opencl/vulkan/hip",
        },
    }
    output = RESULTS_DIR / "coupling_surface.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {output}")
    alive = [name for name, item in result["ports"].items() if item["alive"]]
    print(f"alive_ports={alive}")


if __name__ == "__main__":
    main()
