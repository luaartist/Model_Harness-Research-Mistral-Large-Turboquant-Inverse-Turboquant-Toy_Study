# Experiment 0021: Bare-Metal llama.cpp Coupling

## Purpose

The compression work should not stop at packed artifacts. The machine already
has bridge units, GPU shims, runtime servers, ZLUDA libraries, OpenCL/Vulkan/HIP
surfaces, and a custom llama.cpp rebuild. This experiment maps how to couple the
quantization priors to an actual decode or inference path.

## Existing Surfaces

Bridge units:

- `$BRIDGE_ROOT/unit12_hip_bridge/src/tph_hip_bridge_server.py`
  provides allocation, copy, OpenCL source compilation, module lookup, and kernel
  launch through a HIP-shaped HTTP API on port 8504.
- `$BRIDGE_ROOT/unit12_hip_bridge/src/websocket_gpu_bridge.py`
  bridges status and compute requests to Wine/Cygwin clients.
- `$BRIDGE_ROOT/unit12_hip_bridge/src/gpu_interceptor.py`
  ranks GPU servers on ports 8501-8505 and routes to the best live backend.
- `$BRIDGE_ROOT/unit12_hip_bridge/src/amdhip64_shim.c`
  is a HIP-to-OpenCL shim for Wine-side clients.
- `$BRIDGE_ROOT/unit13_zluda/zluda/`
  contains CUDA-facing ZLUDA libraries.

llama.cpp rebuild:

- `$LLAMA_CPP_DIR/src/llama-quant.cpp`
  is the weight quantization artifact hook.
- `$LLAMA_CPP_DIR/src/llama-kv-cache.cpp`
  is the KV-cache allocation/type hook.
- `$LLAMA_CPP_DIR/ggml/src/ggml-quants.c`
  is the quant/dequant format hook.
- `$LLAMA_CPP_DIR/ggml/src/ggml-opencl/ggml-opencl.cpp`
  is the easiest first backend target because the bridge server already speaks
  OpenCL source.
- `$LLAMA_CPP_DIR/ggml/src/ggml-vulkan/ggml-vulkan.cpp`
  is the shader/pipeline target for a later native backend.
- `$LLAMA_CPP_DIR/ggml/src/ggml-hip/CMakeLists.txt`
  wires HIP through the ggml CUDA template backend.

## Patch Applied

The HIP/OpenCL bridge launch endpoint now supports typed scalar arguments while
remaining backward compatible with plain buffer-handle lists. This matters
because real quant/dequant kernels need sizes, strides, scales, and metadata, not
only buffers.

New launch arg forms:

```json
{"kind": "buffer", "handle": 4096}
{"kind": "i32", "value": 256}
{"kind": "u32", "value": 128}
{"kind": "i64", "value": 1024}
{"kind": "u64", "value": 1024}
{"kind": "size_t", "value": 1024}
{"kind": "f32", "value": 0.03125}
{"kind": "f64", "value": 0.03125}
```

## Coupling Plan

### Phase 0: Sidecar Kernel Probe

Start the bridge server and run a tiny OpenCL quant/dequant kernel through port
8504. Use synthetic vectors first, then 0015/0018/0020 packed shardlets.

Goal: prove the bridge can decode our packed representation without touching
llama.cpp internals.

### Phase 1: llama.cpp Quantizer Hook

Add an experimental export mode in `llama-quant.cpp` that can emit a sidecar
catalog for Theomatica/flavor/TPH stagger choices. Do not introduce a new GGUF
type until byte accounting and decode behavior are stable.

Goal: produce files that a runtime can map, checksum, and decode deterministically.

### Phase 2: KV-Cache Hook

Use `llama-kv-cache.cpp` to test cache type choices and allocation boundaries.
This is the most market-relevant path because TurboQuant, FP8 KV, and INT8 KV
are the competitors.

Goal: compare key inner-product preservation and attention output quality on
real activations.

### Phase 3: Native Backend Kernel

Once the sidecar bridge proves the math, move decode kernels into ggml backend
code. OpenCL is first, Vulkan second, HIP third unless ROCm templates are already
the easiest path on the MI300X setup.

Goal: remove HTTP/sidecar overhead and make the format a real runtime option.

## Current Probe Result

Run:

```bash
python3 probe_coupling_surface.py
```

The first probe wrote `results/coupling_surface.json`. No GPU bridge services
were listening on ports 8501-8505 during that first probe.

## Live Bridge Result

Global packages installed into system `python3`:

```text
pyopencl==2026.1.2
flask==3.1.3
flask-cors==6.0.2
websockets==16.0
```

`pyopencl` sees one AMD OpenCL platform and a `gfx942:sramecc+:xnack-` device
with about 205.8 GB visible memory.

The HIP/OpenCL bridge server started on port 8504:

```text
http://127.0.0.1:8504
device: gfx942:sramecc+:xnack-
VRAM: 205.8 GB
compute units: 304
```

The bridge self-test passed:

```text
status: ALL_PASS
malloc: ok
memcpy_htod: ok
memcpy_dtoh: ok
kernel: add_one match=True
```

The refreshed coupling probe reports `hip_bridge` as the live server. This makes
Phase 0 actionable: submit a small quant/dequant OpenCL kernel through the bridge
before modifying llama.cpp internals.

## Phase 0 Runtime Probe

`run_bridge_shardlet_decode_probe.py` submits an OpenCL kernel to the live bridge
and decodes the MSE packed-code portion of a real q4 shardlet prefix. It uses the
actual `mse_indices` and `norms` tensors from the packed safetensors artifact,
uploads them to GPU memory through the bridge, launches a decode proxy kernel,
copies the output back, and validates it against a CPU mirror.

This is not the full TurboQuant inverse yet. It proves the runtime plumbing for
packed bytes, norms, codebook values, typed scalar kernel args, GPU launch, and
round-trip validation. The next step is to add the full TurboQuant inverse:
MSE codebook decode, inverse rotation, QJL residual sign decode, and residual
projection.

Run result:

```text
script: run_bridge_shardlet_decode_probe.py
artifact: experiments/0018_same_harness_prior_shootout/artifacts/prior_shootout_q4_shardlet.safetensors
prefix: c0.q4.flavor_best
device: gfx942:sramecc+:xnack-
rows: 256
packed_cols: 64
packed_bytes: 16384
output_bytes: 131072
max_abs_diff: 0.0
mean_abs_diff: 0.0
allclose: True
result: results/bridge_shardlet_decode_probe.json
```

Interpretation: real q4 packed shardlet bytes were decoded by an OpenCL kernel
through the HIP-shaped bridge, copied back, and matched the CPU mirror exactly.
This is the first runtime-side coupling proof for the packed artifacts.

## Phase 0B Full TurboQuant Inverse

`run_bridge_turboquant_inverse_probe.py` runs the full q4 TurboQuant inverse
through the same bridge. It uploads the real packed shardlet tensors:

- `mse_indices`
- `qjl_signs`
- `norms`
- `residual_norms`
- `pi`
- Lloyd-Max centroids
- generated QJL matrix `S`

The OpenCL kernel computes:

```text
x_mse[col] = norm[row] * sum_j centroid[mse_code_j] * pi[j, col]
x_qjl[col] = residual_norm[row] * sqrt(pi/2)/d * sum_j sign_j * S[j, col]
x_hat[col] = x_mse[col] + x_qjl[col]
```

It compares the bridge output against the Torch `TurboQuantProd.dequantize`
reference.

Run result:

```text
script: run_bridge_turboquant_inverse_probe.py
artifact: experiments/0018_same_harness_prior_shootout/artifacts/prior_shootout_q4_shardlet.safetensors
prefix: c0.q4.flavor_best
device: gfx942:sramecc+:xnack-
rows: 256
vector_dim: 128
mse_packed_bytes: 16384
qjl_packed_bytes: 4096
pi_bytes: 65536
output_bytes: 131072
max_abs_diff: 2.60770320892334e-08
mean_abs_diff: 1.7736779822641324e-09
rms_diff: 2.52964049707316e-09
allclose_1e_4: True
allclose_1e_5: True
```

Runtime package files:

```text
runtime_package/runtime_package_manifest.json
runtime_package/turboquant_inverse_q4.opencl
```

Interpretation: the bridge can now run the complete q4 TurboQuant inverse over a
real packed shardlet and match the canonical Torch implementation within normal
float32 accumulation noise. This is the packaging boundary for the next step:
move the kernel behind a ggml-opencl sidecar backend or a llama.cpp KV-cache
decode hook.

## Phase 0C Runner Package Contract

The runtime package manifest is now runner-facing rather than lab-only. It
records:

- source artifact path, byte size, and SHA-256
- selected safetensors prefix
- tensor shapes, dtypes, byte sizes, and SHA-256 values
- generated centroid and QJL matrix shapes, byte sizes, and SHA-256 values
- OpenCL kernel path, function name, byte size, and SHA-256
- bridge backend and device metadata
- validation result path and numeric tolerances
- load order, input contract, output contract, and claim boundary

This makes the package auditable before a runner consumes it. A model runner can
reject stale tensors, wrong kernels, wrong seeds, or mismatched package metadata
before launching GPU decode.

Runner package files:

```text
runtime_package/runtime_package_manifest.json
runtime_package/turboquant_inverse_q4.opencl
runtime_package/RUNNER_README.md
runtime_package/validate_runtime_package.py
```

Interpretation: this is the first handoff shape for someone who only wants to
run a model. It does not require them to believe the theory; it gives them a
hashable package, a load sequence, and a validated decode target.

The validator writes `results/runtime_package_validation.json` and lets a runner
preflight the package before GPU decode. This is the practical trust boundary:
if the hashes, schemas, generated matrices, or kernel do not match the manifest,
the runtime should refuse to launch.

Preflight result:

```text
script: runtime_package/validate_runtime_package.py
result: results/runtime_package_validation.json
valid: True
failures: 0
tensor_count: 5
generated_input_count: 2
stored_tensor_bytes: 88064
generated_matrix_bytes: 65568
```

The validator caught and resolved an important packaging distinction: the source
safetensors file stores some tensors in lower precision, while the runtime
kernel consumes `float32` inputs after loader conversion. The manifest now makes
that conversion explicit and hashes the runtime upload form.

## Phase 0D QuantSIM Runner Adapter

The package is now reachable from the source-only `QuantSIM-runtime` connector
system through a tracked plugin:

```text
$QUANTSIM_RUNTIME/src/quantsim_runtime/connectors/plugins/shardlet_runtime
```

The plugin exposes a small runner-facing preflight API:

```text
op=validate          checks package artifact/kernel records
op=probe_routes     probes GPU interceptor / HIP bridge routes
op=runner_contract  returns package load/input/output contract
op=preflight        validates package and selects best live route
```

The QuantSIM job queue also has a one-action template:

```bash
PYTHONPATH=src /usr/bin/python3 -m quantsim_runtime.jobs.cli trigger --template shardlet_runtime_preflight
```

Run result:

```text
state: done
package valid: True
best_route: http://127.0.0.1:8504
backend: HIP_Bridge_v1_OpenCL_rusticl
device: gfx942:sramecc+:xnack-
vram_total_gb: 205.8
```

Interpretation: the package now has a bridge between lab artifacts and a general
runtime job system. This is still pre-inference, but it is the first practical
runner adapter: a user can ask the runtime whether the package is trustworthy
and whether the AMD decode route is alive before attempting model execution.

## Phase 0E GGML/OpenCL Sidecar

`runtime_package/ggml_opencl_sidecar.py` is the first llama.cpp-adjacent decode
adapter. It deliberately stays outside llama.cpp core while the package shape is
still stabilizing.

The sidecar:

- consumes `runtime_package_manifest.json`
- reruns package validation before launch
- loads packed tensors from the safetensors artifact
- generates Lloyd-Max centroids and the QJL matrix from manifest parameters
- compiles and launches `turboquant_inverse_q4` through the AMD bridge
- writes contiguous row-major `float32 [rows, vector_dim]` output
- writes JSON metadata with bridge status, output byte count, SHA-256, and sample

Smoke run:

```bash
cd runtime_package
/usr/bin/python3 ggml_opencl_sidecar.py --rows 8
```

Default outputs:

```text
../results/ggml_opencl_sidecar_decode.f32
../results/ggml_opencl_sidecar_decode.json
```

Full sidecar run result:

```text
rows: 256
vector_dim: 128
output_bytes: 131072
output_sha256: 0213eed45ca2b664f4979228ffdb332eb0624734daf717012016df6c76d1a2a8
bridge: http://127.0.0.1:8504
backend: HIP_Bridge_v1_OpenCL_rusticl
device: gfx942:sramecc+:xnack-
memory_handles_after_decode: 0
```

Interpretation: the next target is no longer "create a sidecar". It is now a
native GGML/GGUF loader shim that maps this validated sidecar output into a
`ggml_tensor` buffer and eventually replaces the HTTP bridge with an in-process
backend path.

## Phase 0F GGML Tensor Buffer Shim

`runtime_package/ggml_tensor_buffer_shim.py` consumes the sidecar metadata and
the decoded `.f32` bytes, verifies byte count, SHA-256, dtype, shape, and sample,
then emits a GGML-style tensor descriptor.

Descriptor output:

```text
../results/ggml_tensor_buffer_shim.json
```

Current mapping:

```text
tensor: c0.q4.flavor_best.decoded_f32
ggml_type: GGML_TYPE_F32
shape_row_major: [256, 128]
ggml_ne: [128, 256, 1, 1]
ggml_nb: [4, 512, 131072, 131072]
nbytes: 131072
```

Interpretation: this is close to the gated-runtime pattern: the expensive build
artifact is held behind hashes and a manifest; the public runtime layer verifies
the package, maps bytes into the expected tensor layout, and exits. The remaining
private-runner work is to replace the JSON descriptor with a real C/C++ call to
`ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, 256)` and a backend upload or
`memcpy` into `tensor->data`.

## Phase 0G Private Runner C Shim

A local Grok scaffolding probe recommended keeping the next layer external:
validate package, decode if needed, map the f32 block into a GGML-like tensor
allocation, emit audit metadata, and keep the backend interface replaceable.

`runtime_package/private_runner_shim/` now implements the first compileable C
version of that handoff:

```text
Makefile
ggml_tq_tensor_shim.h
ggml_tq_tensor_shim.c
smoke_ggml_tq_tensor_shim.c
```

The shim validates:

```text
rows: 256
cols: 128
ggml_type: GGML_TYPE_F32
ggml_ne: [128, 256, 1, 1]
ggml_nb: [4, 512, 131072, 131072]
nbytes: 131072
finite_values: enforced
```

Build and smoke run:

```bash
cd runtime_package/private_runner_shim
make clean && make
./smoke_ggml_tq_tensor_shim \
  ../../results/ggml_opencl_sidecar_decode.f32 \
  ../../results/private_runner_api_audit.json \
  decoded_f32_file
```

Validated result:

```text
build: clean, no warnings
adapter: private-runner-c-shim
backend: decoded_f32_file
l2_norm: 2.0209344811802672
audit: results/private_runner_api_audit.json
```

Interpretation: the package now has a native runner boundary, still outside
llama.cpp core. The next implementation target is to drive that tensor through
the llama.cpp model loader's named-tensor path while keeping the HTTP route as a
separate backend until an in-process OpenCL/HIP backend is warranted.

## Phase 0H Private Runner API Boundary

The C scaffold now exposes a request/result API:

```text
tq_runner_decode(request, result, err, err_cap)
tq_runner_result_free(result)
```

Supported backends today:

```text
decoded_f32_file
http_bridge
```

Reserved backend names:

```text
inprocess_opencl
```

Validation:

```text
decoded_f32_file exit: 0
http_bridge exit: 0 when the bridge is online
http_bridge mode: live_decode
```

Interpretation: this moves the handoff from a one-off smoke executable to a
small private-runner API. The C `http_bridge` backend now invokes the same
decode flow previously owned only by `ggml_opencl_sidecar.py`. The remaining
backend question is whether to keep HTTP as the integration boundary or add an
`inprocess_opencl` backend that removes HTTP entirely.

## Phase 0I Python Framework Notes

`runtime_package/runner_framework.py` is now the first-build orchestrator. This
keeps the moving framework logic in Python before a later Rust port.

The driver runs:

```text
live bridge /status probe
validate_runtime_package.py
ggml_tensor_buffer_shim.py
private_runner_shim make clean
private_runner_shim make
private runner decoded_f32_file API smoke
real GGML loader-surface smoke
private runner http_bridge live decode when the bridge is online
native-vs-Python sidecar SHA-256 comparison
```

Output:

```text
results/python_runner_framework_audit.json
```

Validated result:

```text
adapter: python-runner-framework
package_valid: true
tensor: c0.q4.flavor_best.decoded_f32
ggml_ne: [128, 256, 1, 1]
ggml_nb: [4, 512, 131072, 131072]
decoded_f32_file: implemented
http_bridge: live_decode
native-vs-sidecar: byte-identical
ggml_loader_surface: real ggml_tensor, byte-identical copy
```

## Phase 0M Real GGML Loader Surface

The private runner now has a GGML-facing adapter:

```text
ggml_tq_loader_adapter.h
ggml_tq_loader_adapter.c
smoke_ggml_tq_loader_adapter.c
```

The smoke links against the local llama.cpp GGML build and materializes the
validated decoded f32 block into `struct ggml_tensor` via:

```text
ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, rows)
memcpy(tensor->data, runner.tensor.data, runner.tensor.nbytes)
```

Validation:

```text
ggml_type: f32
ggml_ne: [128, 256, 1, 1]
ggml_nb: [4, 512, 131072, 131072]
byte_identity: true
release_tag: v0.3-ggml-loader-surface
```

Interpretation: this is still not a full llama.cpp loader patch, but it removes
the last simulated part of the tensor handoff. The decoded TurboQuant output is
now proven to fit a real GGML tensor allocation with no transpose or stride
ambiguity.

Live bridge telemetry in the same audit:

```text
bridge_url: http://127.0.0.1:8504/status
backend: HIP_Bridge_v1_OpenCL_rusticl
device: gfx942:sramecc+:xnack-
compute_units: 304
vram_total_gb: 205.8
memory_handles: 0
vram_allocated: 0
memory_clean: true
```

Interpretation of the bridge log: the Python sidecar, framework, and native C
`http_bridge` backend use the HTTP bridge for real GPU package decode, including
module load, malloc, uploads, kernel launch, download, and free. The repeated
return to `total=0` and the audit's `memory_clean: true` are the important
cleanup signal.

The audit includes framework notes for the future Rust version:

```text
DecodeBackend trait
Result<TensorBuffer, Error>
serde manifest records
sha2 verification
explicit unsupported backend errors until each backend is real
```

Interpretation: Python now owns the first-build automation and the design notes.
Rust should come after the decode contract stops shifting, with the Python audit
as the behavior reference.

## Phase 0J Package Boundary And Factory Scope

The project now distinguishes the build factory from the runner package.

New package-boundary files:

```text
runtime_package/PACKAGE_BOUNDARY.md
runtime_package/PACKAGING_BOUNDARY.json
runtime_package/MODEL_PROFILE_TEMPLATE.json
```

Runner package includes:

```text
manifest and hashes
decode kernel
validators
Python sidecar/framework
native C handoff shim
model profile template
package boundary spec
readme and audit outputs
embedded or externally hash-pinned artifacts
```

Factory-only scope:

```text
full Mistral source shards
streaming slice readers
calibration windows
Theomatica/TPH prior search
TurboQuant fit/search experiments
random/default controls
held-out metrics
pretraining or quantization-aware training experiments
large intermediates and failed candidates
```

Expansion path:

```text
single shardlet -> more rows -> more tensor families -> head_dim 192 and KV
tests -> full layer package -> full model package catalog -> adjacent models
```

Interpretation: the full Mistral build should stay in the factory until the
package catalog is complete. Other models should reuse the same factory profile
only after the Mistral profile can emit repeated shardlet packages with metrics.
Training-time quantization should remain a separate factory research branch
until it can produce frozen artifacts with the same audit boundary.

## Phase 0K v0.1 Shardlet Release Lock

`runtime_package/` now has a package-local release tag:

```text
v0.1-shardlet-q4
```

Release files:

```text
runtime_package/VERSION
runtime_package/Makefile
runtime_package/release_lock.py
runtime_package/RELEASE_NOTES.md
runtime_package/RELEASE_LOCK.v0.1-shardlet-q4.json
```

The verify target is:

```bash
cd runtime_package
make verify
```

It runs package validation, Python sidecar bridge decode, GGML descriptor
generation, private C runner smoke, stable Python framework audit, and release
hash-lock checking.

The release lock snapshots SHA-256 values for:

```text
artifact
kernel
manifest
descriptor
sidecar metadata and f32 output
runtime validation audit
private runner API audit
Python framework audit
package boundary/profile files
release notes
```

Interpretation: this freezes the current one-shardlet proof as a stable baseline
before catalog expansion or native C/Rust bridge work.

## Phase 0L Math-Gated Catalog Scaling

The next scaling phase is not merely "emit more shardlets." It should promote
5 to 10 same-family q4 shardlets only after they pass mathematical gates that
are stricter than the v0.1 decode proof.

New planning/check files:

```text
SCALING_MATH_GATES.md
scaling_math_gate_checks.wl
results/scaling_math_gate_checks.json
```

The catalog target remains:

```text
variant family: c*.q4.flavor_best
vector_dim: 128
decode kernel: runtime_package/turboquant_inverse_q4.opencl
runner catalog: runtime_package/package_catalog.json
runner command: runner_framework.py --catalog package_catalog.json
```

Required promotion gates:

```text
payload/source/runtime byte accounting
heldout reconstruction error
heldout Gram and query-score preservation
spectral and covariance distortion
effective-rank and anisotropy coverage
default/random control comparison
calibration-to-heldout overfit check
same-kernel runtime identity check
```

Wolfram Engine is used for exact byte-accounting checks and row-count threshold
tables. Grok can review the gate list skeptically, but hard thresholds should be
derived from the existing 0015/0018 distributions and controls rather than copied
from a model suggestion.

Interpretation: catalog scaling is the mathematical stress test between the
single-shardlet release and the native C `http_bridge` backend. If the same
kernel and runner contract survive diverse spectra, larger row counts, and
control comparisons, then C/Rust backend work has a stable target.
