# Package Boundary and Expansion Plan

This project now has two different products that should not be mixed together:

```text
factory: builds and validates quantized packages from large source models
runner package: ships only what a runtime needs to verify and decode artifacts
```

## What Ships In A Runner Package

The runner package should be small, deterministic, and auditable. It includes:

```text
runtime_package_manifest.json
turboquant_inverse_q4.opencl
VERSION
Makefile
release_lock.py
RELEASE_NOTES.md
RELEASE_LOCK.v0.3-ggml-loader-surface.json
validate_runtime_package.py
ggml_opencl_sidecar.py
ggml_tensor_buffer_shim.py
runner_framework.py
private_runner_shim/
PACKAGING_BOUNDARY.json
MODEL_PROFILE_TEMPLATE.json
RUNNER_README.md
```

It also needs one of these artifact delivery modes:

```text
embedded artifact: include the packed safetensors shardlet in the package
external artifact: keep an absolute or URI path plus bytes and SHA-256
registry artifact: fetch by model id, prefix, package id, bytes, and SHA-256
```

For the current lab package, the artifact is external and hash-pinned:

```text
artifacts/prior_shootout_q4_shardlet.safetensors
sha256: 65a80313464377cc49965984871aa70c511c08f724d7fa89c50aca7814357092
```

## What Stays In The Factory

The factory stays on this machine while the full Mistral build is still being
created. It owns expensive, unstable, or model-specific material:

```text
full Mistral source shards
streaming slice readers and calibration windows
Theomatica and TPH prior selection code
TurboQuant training / fitting experiments
codebook and seed search experiments
fair random/default controls
held-out quality metrics
pretraining or quantization-aware training experiments
large intermediate tensors and failed candidates
```

The factory can emit many runner packages, but runner packages should not need
the factory to execute a validated decode.

## Expansion Order

1. Current proof: one q4 shardlet, 256 rows by 128 dims.
2. Same tensor family, more rows, same decode contract.
3. More tensor families from Mistral, still shardlet-by-shardlet.
4. Head-dim 192 and attention/KV-cache shaped tests.
5. Full layer package with multiple prefixes and tensor descriptors.
6. Full model package catalog with many shardlets and loader routing.
7. Repeat the factory profile on adjacent Mistral-like models.
8. Only then consider quantization-aware pretraining or training-time quantized
   checkpoints as a separate factory research branch.

## Is It Just Code Right Now?

Not anymore. The current package includes code plus a manifest, hashes, bridge
telemetry, decoded output hashes, native tensor descriptors, and audit files.
However, it is still a shardlet package rather than a full model package.

The full product shape should become:

```text
model profile
package manifest
packed artifacts
decode kernels
validator
runner framework
native handoff layer
audits
claim boundary
```

## Pretraining In Quantization

Pretraining or quantization-aware training should stay in the factory for now.
It changes model weights and therefore changes the claim boundary. Runner
packages should consume frozen artifacts. If training-time quantization proves
useful, it should produce a new package family with separate manifests and
comparisons against post-training quantization.

## Next Packaging Step

Create a package catalog that can hold multiple shardlets for the same model:

```text
package_id
model_id
tensor_family
prefix
artifact_uri
artifact_sha256
kernel_id
decode_contract
runner_tensor_descriptor
quality_metrics
claim_boundary
```

That catalog is the bridge from a lab proof to a model runner experience.
