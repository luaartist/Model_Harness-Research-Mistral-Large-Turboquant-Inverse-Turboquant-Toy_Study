# Private Runner Shim

This directory is the first native handoff scaffold for the verified q4 shardlet
runtime package. It stays outside llama.cpp core and can either consume a
decoded row-major `.f32` output produced by `ggml_opencl_sidecar.py` or decode a
TurboQuant shardlet through the live HIP/OpenCL HTTP bridge.

## Build

```bash
make clean && make
```

The native bridge backend links against libcurl. The Makefile uses `pkg-config`
when available and falls back to `-lcurl`.

## Smoke Run

```bash
./smoke_ggml_tq_tensor_shim \
  ../../results/ggml_opencl_sidecar_decode.f32 \
  ../../results/private_runner_api_audit.json \
  decoded_f32_file
```

To exercise the native HTTP bridge path against the runtime package manifest:

```bash
./smoke_ggml_tq_tensor_shim \
  ../../results/ggml_opencl_sidecar_decode.f32 \
  ../../results/private_runner_http_bridge_audit.json \
  http_bridge \
  ../runtime_package_manifest.json \
  http://127.0.0.1:8504
```

The smoke executable validates the native tensor descriptor:

```text
ggml_type: GGML_TYPE_F32
rows: 256
cols: 128
ggml_ne: [128, 256, 1, 1]
ggml_nb: [4, 512, 131072, 131072]
nbytes: 131072
```

It writes an audit JSON with the tensor contract, L2 norm, sample values, and the
expected SHA-256 from the Python descriptor layer. SHA-256 enforcement remains in
`validate_runtime_package.py` and `ggml_tensor_buffer_shim.py`; this C shim is
for native layout validation and runner handoff.

## Runner API Boundary

The public C surface now routes through `tq_runner_decode()`. The implemented
backends are:

```text
decoded_f32_file
http_bridge
```

Reserved backend names still fail explicitly:

```text
inprocess_opencl
```

The `http_bridge` backend parses the package or catalog manifest, reads the
safetensors shardlet directly, converts F16 metadata tensors to F32, uploads all
decode inputs through the HIP/OpenCL bridge, launches `turboquant_inverse_q4`,
downloads the decoded F32 tensor, and writes the same runner audit contract as
the file backend.
