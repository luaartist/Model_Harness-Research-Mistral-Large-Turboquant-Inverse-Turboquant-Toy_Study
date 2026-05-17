# Model Harness

Small, auditable harness for validating a TurboQuant-style q4 shardlet decode path and GGML-style handoff for Mistral Large 3 research artifacts.

This repository is a runner/package surface, not a full model dump. It includes source, manifests, release locks, generated deterministic inputs, and native shim code. Full model shards, GGUF outputs, safetensor artifacts, decode dumps, private logs, and machine-local workspaces are intentionally excluded.

## Current Claim Boundary

This harness demonstrates:

- q4 shardlet manifest validation by bytes and SHA-256
- OpenCL decode kernel wiring for packed q4 inputs
- deterministic generated inputs for centroids and QJL matrix checks
- GGML-style tensor descriptor generation
- C shim source for native decoded-f32 handoff and GGML loader-surface smoke tests

It does not claim:

- full Mistral 675B inference
- compression superiority over GGUF/GPTQ/AWQ baselines
- native GGUF quant type support
- public redistribution rights for model-derived artifacts

## Repository Layout

```text
runtime_package/            runner-facing source, manifests, release locks
runtime_package/generated_inputs/
                            small deterministic raw inputs tracked by hash
runtime_package/private_runner_shim/
                            C source for decoded-f32 and GGML handoff smoke tests
docs/                       method, catalog, and architecture notes
research_tools/             optional lab/factory probes, not required by runner checks
schemas/                    lightweight JSON schemas for manifest validation
```

## Artifact Policy

Model-derived artifacts are referenced by logical path, bytes, and SHA-256, but are not committed. Provide them through a private artifact store, GitHub Release asset, Hugging Face artifact, or local `artifacts/` directory before running full validation.

The primary demo artifact expected by the current manifest is:

```text
artifacts/prior_shootout_q4_shardlet.safetensors
sha256: 65a80313464377cc49965984871aa70c511c08f724d7fa89c50aca7814357092
```

## Environment

Install Python dependencies:

```bash
python -m pip install -r requirements.txt
```

Optional environment variables:

```bash
export MODEL_LAB=/path/to/Mistral-Large-Quantization-Project/_model_lab
export MODEL_DIR=/path/to/Mistral-Large-Quantization-Project
export TURBOQUANT_ROOT=/path/to/0xSero_turboquant
export GGUF_PY_PATH=/path/to/llama.cpp/gguf-py
export LLAMA_CPP_DIR=/path/to/llama.cpp
export BRIDGE_URL=http://127.0.0.1:8504
```

## Quick Checks

Syntax-check the Python sources:

```bash
python -m py_compile runtime_package/*.py research_tools/*.py
```

Validate the runtime manifest schema:

```bash
python -m json.tool runtime_package/runtime_package_manifest.json >/dev/null
```

Run full package validation only after providing the hash-pinned artifact and optional bridge dependencies:

```bash
python runtime_package/validate_runtime_package.py
```

## Provenance

The method references TurboQuant concepts, llama.cpp/GGUF conventions, Mistral-family model artifacts, and an AMD HIP/OpenCL bridge environment. See [NOTICE.md](NOTICE.md) and [docs/METHODS.md](docs/METHODS.md) before redistributing artifacts or claiming model compatibility.

## License

All rights reserved. See [LICENSE](LICENSE).