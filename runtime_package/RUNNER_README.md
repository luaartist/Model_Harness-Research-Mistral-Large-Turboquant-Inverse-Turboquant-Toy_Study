# Runtime Package: Mistral Large q4 Shardlet

This package is a runner-facing proof that one real packed q4 shardlet can be
decoded by the AMD bridge runtime and checked against the canonical Torch
TurboQuant inverse.

## What This Is

The package binds together:

- a safetensors shardlet artifact
- a selected prefix inside that artifact
- an OpenCL decode kernel
- generated Lloyd-Max centroids
- a generated QJL projection matrix
- hashes and byte accounting for the runtime inputs
- validation against `TurboQuantProd.dequantize`

The runner-facing contract is simple:

```text
packed q4 shardlet + manifest + OpenCL kernel -> float32 decoded vectors
```

## Release Tag

```text
v0.3-ggml-loader-surface
```

The public source package is file-locked by `RELEASE_LOCK.v0.3-ggml-loader-surface.json`. This is a
package-local release lock, not a git tag.

## Current Verified Target

```text
artifact: artifacts/prior_shootout_q4_shardlet.safetensors
prefix: c0.q4.flavor_best
kernel: turboquant_inverse_q4.opencl
device: gfx942:sramecc+:xnack-
rows: 256
vector_dim: 128
output: float32 [256, 128]
```

The latest validation matched the Torch reference with `allclose_1e_5=True` and
`max_abs_diff=2.60770320892334e-08`.

## Files

```text
runtime_package_manifest.json
turboquant_inverse_q4.opencl
VERSION
Makefile
release_lock.py
RELEASE_NOTES.md
RELEASE_LOCK.v0.3-ggml-loader-surface.json
RUNNER_README.md
ggml_opencl_sidecar.py
ggml_tensor_buffer_shim.py
runner_framework.py
private_runner_shim/
PACKAGE_BOUNDARY.md
PACKAGING_BOUNDARY.json
MODEL_PROFILE_TEMPLATE.json
```

## Load Order

1. Read `runtime_package_manifest.json`.
2. Verify the safetensors artifact hash.
3. Open the safetensors artifact and select tensors by `prefix`.
4. Verify tensor shapes, dtypes, byte sizes, and hashes.
5. Apply manifest-declared loader conversions before upload.
6. Load hash-pinned generated centroid and QJL inputs from `generated_inputs/`.
7. Compile `turboquant_inverse_q4.opencl`.
8. Upload tensors and generated inputs to the AMD bridge runtime.
9. Launch `turboquant_inverse_q4`.
10. Read back `float32 [rows, vector_dim]` decoded vectors.

## Preflight Validation

Before launching GPU decode, run:

```bash
/usr/bin/python3 validate_runtime_package.py
```

This validates the artifact hash, kernel hash, runtime tensor shapes, runtime
tensor hashes after loader conversion, generated centroid hash, and generated
QJL matrix hash. It writes the result to:

```text
../results/runtime_package_validation.json
```

## Reproducible Verify

Run the full verification target from `runtime_package/`:

```bash
make verify
```

The target runs:

```text
validate_runtime_package.py
ggml_opencl_sidecar.py
ggml_tensor_buffer_shim.py
private_runner_shim C build and decoded_f32_file smoke
runner_framework.py --stable-audit
release_lock.py --check
```

The public release lock snapshots SHA-256 values for the source package,
generated inputs, kernel, manifest, package catalog, and external artifact hash
record. Live decode audits are generated locally under `../results/` and are not
committed.

## GGML/OpenCL Sidecar Decode

The first llama.cpp-adjacent adapter is an external sidecar, not a patch to
llama.cpp core. It consumes the manifest, validates the package, launches the
OpenCL inverse through the AMD bridge, and writes row-major raw `float32` output
that a later GGML/GGUF loader shim can map into a tensor buffer.

Run a smoke decode of the first few rows:

```bash
/usr/bin/python3 ggml_opencl_sidecar.py --rows 8
```

Run the full current shardlet:

```bash
/usr/bin/python3 ggml_opencl_sidecar.py
```

Default outputs:

```text
../results/ggml_opencl_sidecar_decode.f32
../results/ggml_opencl_sidecar_decode.json
```

The `.f32` file is contiguous row-major `float32 [rows, vector_dim]`; the JSON
sidecar records the package, bridge route, output byte count, SHA-256, and a
small decoded sample.

Verified full-shardlet sidecar result:

```text
rows: 256
vector_dim: 128
output_bytes: 131072
output_sha256: 0213eed45ca2b664f4979228ffdb332eb0624734daf717012016df6c76d1a2a8
bridge: http://127.0.0.1:8504
device: gfx942:sramecc+:xnack-
```

## GGML Tensor Buffer Shim

After the sidecar writes raw `float32`, the tensor shim validates those bytes and
emits GGML-style dimensions and strides for a future native loader.

```bash
/usr/bin/python3 ggml_tensor_buffer_shim.py
```

Default output:

```text
../results/ggml_tensor_buffer_shim.json
```

The descriptor maps the decoded block as:

```text
name: c0.q4.flavor_best.decoded_f32
ggml_type: GGML_TYPE_F32
shape_row_major: [256, 128]
ggml_ne: [128, 256, 1, 1]
ggml_nb: [4, 512, 131072, 131072]
nbytes: 131072
```

This is the same pattern as a gated runtime build: the artifact is fixed and
hash-checked; the software layer only verifies, maps, and hands off the buffer.

## Private Runner C Shim

`private_runner_shim/` is a compileable C scaffold for the next automation layer.
It does not patch llama.cpp core. It loads the sidecar `.f32` output, enforces
the GGML-style shape and stride contract, checks values are finite, and writes a
runner audit JSON.

Build and run:

```bash
cd private_runner_shim
make clean && make
./smoke_ggml_tq_tensor_shim \
  ../../results/ggml_opencl_sidecar_decode.f32 \
  ../../results/private_runner_api_audit.json \
  decoded_f32_file
```

Verified output:

```text
adapter: private-runner-c-shim
backend: decoded_f32_file
ggml_ne: [128, 256, 1, 1]
ggml_nb: [4, 512, 131072, 131072]
nbytes: 131072
l2_norm: 2.0209344811802672
audit: ../results/private_runner_api_audit.json
```

The Python preflight and descriptor layer remain the SHA-256 gate. The C shim is
the native handoff surface: it proves the decoded bytes can become a contiguous
GGML-compatible tensor buffer without transpose or layout ambiguity.

## Real GGML Loader Surface Smoke

`private_runner_shim/ggml_tq_loader_adapter.c` takes the validated runner
buffer and materializes it into an actual `struct ggml_tensor` from the local
llama.cpp rebuild. The smoke allocates `ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
cols, rows)`, copies the decoded bytes into `tensor->data`, checks GGML shape,
strides, contiguity, and byte identity, then writes
`../results/ggml_loader_surface_audit.json`.

Build and run:

```bash
cd private_runner_shim
make clean && make
./smoke_ggml_tq_loader_adapter \
  ../../results/ggml_opencl_sidecar_decode.f32 \
  ../../results/ggml_loader_surface_audit.json \
  decoded_f32_file
```

Verified output:

```text
backend: decoded_f32_file
ggml_type: f32
ggml_ne: [128, 256, 1, 1]
ggml_nb: [4, 512, 131072, 131072]
byte_identity: true
```

The C API now supports two concrete handoff paths:

```text
decoded_f32_file
http_bridge
```

`decoded_f32_file` maps the predecoded sidecar bytes. `http_bridge` parses the
manifest, reads the safetensors shardlet directly, uploads all TurboQuant decode
inputs through the HIP/OpenCL bridge, launches `turboquant_inverse_q4`, and
downloads a contiguous GGML-compatible F32 tensor.

When the bridge is online, the framework also dumps the native `http_bridge`
runtime output and verifies its SHA-256 matches the Python sidecar output.

## Python Runner Framework

`runner_framework.py` is the first-build orchestration layer. It keeps the
workflow in Python while the package format is still moving, and writes
framework notes for the later Rust port.

Run:

```bash
/usr/bin/python3 runner_framework.py
```

Default output:

```text
../results/python_runner_framework_audit.json
```

The framework run does the full current handoff:

```text
live bridge /status probe
validate_runtime_package.py
ggml_tensor_buffer_shim.py
make clean && make in private_runner_shim
private runner decoded_f32_file API smoke
real GGML loader-surface smoke
private runner http_bridge live decode when the bridge is online
native-vs-Python sidecar SHA-256 comparison
```

The live bridge probe records the current Python/HTTP decode route:

```text
bridge: http://127.0.0.1:8504/status
backend: HIP_Bridge_v1_OpenCL_rusticl
device: gfx942:sramecc+:xnack-
compute_units: 304
vram_total_gb: 205.8
memory_handles: 0
vram_allocated: 0
memory_clean: true
```

That means the bridge logs are real package activity and cleanup evidence. The
native C `http_bridge` backend now exercises the same bridge route from C for
the runtime package and promoted catalog shardlets.

The audit records the Rust-port sketch as structured `framework_notes`: use a
`DecodeBackend` trait, serde manifests, `sha2` verification, explicit
unsupported backend errors, and one backend implementation at a time.

## Package Boundary

This is no longer only code. The package now has a formal boundary:

```text
factory: builds and validates quantized artifacts from source models
runner package: verifies, decodes, maps, and audits frozen artifacts
```

See:

```text
PACKAGE_BOUNDARY.md
PACKAGING_BOUNDARY.json
MODEL_PROFILE_TEMPLATE.json
```

The factory stays on this machine for full Mistral while the build is still
expanding. It owns source shards, streamed calibration, prior search, controls,
held-out metrics, and any quantization-aware training experiments. Runner
packages should ship only frozen artifacts or hash-pinned artifact references,
decode kernels, manifests, validators, runner wiring, and audits.

## Claim Boundary

This is a validated shardlet runtime package. It is not yet a full Mistral Large
inference runner, a native GGUF/GGML quant type, or a compression-superiority
claim against market baselines.

The next practical target is to map the decoded tensor buffer into the
llama.cpp model loader path for named tensors while keeping the native
`http_bridge` route as the decode backend behind that loader boundary.
