# Research Tools

This directory contains optional lab/factory probes used to build and inspect the harness. They are not required for the public runner package checks.

These tools may require local model shards, flavor artifacts, a TurboQuant checkout, Wolfram, a custom llama.cpp build, and a live AMD bridge service. Configure paths with environment variables before running:

```bash
export MODEL_DIR=/path/to/Mistral-Large-Quantization-Project
export MODEL_LAB=/path/to/Mistral-Large-Quantization-Project/_model_lab
export FLAVOR_PATH=/path/to/flavor.safetensors
export BRIDGE_ROOT=/path/to/bridge_units
export LLAMA_ROOT=/path/to/llama.cpp
export BRIDGE_URL=http://127.0.0.1:8504
```

The runner package boundary lives in `runtime_package/`. Do not commit generated artifacts from these tools unless they are explicitly added to the public artifact policy by hash and license review.