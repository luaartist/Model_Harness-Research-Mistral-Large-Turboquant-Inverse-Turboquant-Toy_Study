# Architecture

The public harness boundary is intentionally small:

```text
artifact URI + manifest + generated inputs + OpenCL kernel
        -> validator
        -> sidecar decode
        -> GGML-style descriptor
        -> native C handoff surface
```

The factory remains outside this repository. Factory-only material includes full Mistral shards, calibration windows, prior searches, rejected candidates, training experiments, and large intermediates.

## Runner Components

- `runtime_package/runtime_package_manifest.json`: byte/hash contract for one q4 shardlet decode package.
- `runtime_package/turboquant_inverse_q4.opencl`: decode kernel for the current q4 representation.
- `runtime_package/validate_runtime_package.py`: manifest and tensor integrity validator.
- `runtime_package/ggml_opencl_sidecar.py`: local bridge-side decode driver.
- `runtime_package/ggml_tensor_buffer_shim.py`: GGML-style shape/stride descriptor generation.
- `runtime_package/private_runner_shim/`: C source for decoded-f32 and GGML loader-surface smoke tests.

## Configuration

Machine-local paths are configured by environment variables instead of being part of the public release identity:

- `MODEL_LAB`
- `MODEL_DIR`
- `TURBOQUANT_ROOT`
- `GGUF_PY_PATH`
- `LLAMA_CPP_DIR`
- `BRIDGE_URL`

Release identity should be logical names plus bytes and SHA-256 values, not workstation paths.