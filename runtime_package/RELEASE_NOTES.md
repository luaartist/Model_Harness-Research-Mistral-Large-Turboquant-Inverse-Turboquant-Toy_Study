# v0.3-ggml-loader-surface Release Notes

This release locks the first real GGML loader-surface smoke for the
runner-facing q4 TurboQuant shardlet package.

## What Works

```text
package manifest validation
artifact, kernel, tensor, and generated-input hash checks
Python HTTP bridge decode through gfx942 / OpenCL rusticl
row-major float32 sidecar output
GGML-style tensor descriptor generation
private C runner API for decoded_f32_file handoff
native C http_bridge decode through the HIP/OpenCL bridge
native-vs-Python sidecar byte identity checks
real GGML `ggml_new_tensor_2d` materialization smoke
Python framework audit with live bridge status
catalog decode verification for promoted shardlets
release hash lock for the public v0.3 loader-surface source package
```

Verified tensor:

```text
tensor: c0.q4.flavor_best.decoded_f32
shape_row_major: [256, 128]
ggml_ne: [128, 256, 1, 1]
ggml_nb: [4, 512, 131072, 131072]
nbytes: 131072
sidecar_output_sha256: 0213eed45ca2b664f4979228ffdb332eb0624734daf717012016df6c76d1a2a8
```

## What Remains Reserved

```text
inprocess_opencl backend: reserved, not implemented
full Mistral inference: outside this release
native GGUF/GGML quant type: outside this release
full llama.cpp loader integration: outside this release
```

## GGML Loader Surface Checkpoint

```text
loader smoke: pass
source backend: decoded_f32_file
ggml_type: f32
ggml_ne: [128, 256, 1, 1]
ggml_nb: [4, 512, 131072, 131072]
byte_identity: true
audit: results/ggml_loader_surface_audit.json
```

## Native Bridge Checkpoint

```text
runtime http_bridge decode: pass
catalog promoted entries: 4/4 integrity, 4/4 native decode
native-vs-sidecar comparison: 4/4 catalog entries
backend: HIP_Bridge_v1_OpenCL_rusticl
```

## Claim Boundary

This release proves package decode correctness and runner handoff for one real
q4 shardlet. It does not prove full 675B inference, native GGUF loading,
compression superiority over market baselines, or training-time quantization
benefit.

## Verification

Check the public package lock from `runtime_package/`:

```bash
python release_lock.py --check
```

Full bridge verification is still available with `make verify` after the
hash-pinned artifact, TurboQuant reference code, llama.cpp build, and local
bridge endpoint are configured. Those live artifacts are generated locally and
are not part of the public source lock.
