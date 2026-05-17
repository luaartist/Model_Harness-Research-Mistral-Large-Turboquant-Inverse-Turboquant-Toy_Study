# Methods

Quantization methodology for Mistral-Large-3-675B on AMD MI300X.

## 1. TurboQuant q4 Vector Quantization

Adapted from [0xSero/turboquant](https://github.com/0xSero/turboquant), originally a
KV-cache compression scheme. We apply its encode/decode pipeline to static model
weight slices streamed from 272 safetensor shards (~635 GB total).

### 1.1 Encode Pipeline

Given a weight matrix $W \in \mathbb{R}^{n \times d}$ with $d = 128$ (vector_dim):

**Step 1 — Random rotation.**
Generate a random orthogonal matrix $\Pi \in \mathbb{R}^{d \times d}$ (seed 42).
Rotate each row: $\tilde{w}_i = w_i \Pi$.

**Step 2 — Norm extraction.**
Extract per-row norms $\|w_i\|$ and residual norms. Normalize:
$\hat{w}_i = \tilde{w}_i / \|\tilde{w}_i\|$.

**Step 3 — Lloyd-Max scalar quantization.**
For each element of $\hat{w}_i$, quantize to $2^b$ centroids ($b = 3$, mse_bits)
using the Lloyd-Max algorithm. Store as `mse_indices` (uint8, $n \times d/2$
nibble-packed).

**Step 4 — QJL residual sign projection.**
Generate a Johnson-Lindenstrauss projection matrix $S \in \mathbb{R}^{d \times d}$
(seed 1042). Compute the sign bits of $S \hat{w}_i$ to preserve key inner products.
Store as `qjl_signs` (uint8, $n \times d/8$ bit-packed).
Scale factor: $\sigma = 1/\sqrt{d}$ (qjl_scale = 0.00979...).

**Step 5 — Pack.**
Output safetensors artifact with tensors:
- `{prefix}.mse_indices` — uint8 [$n, d/2$]
- `{prefix}.qjl_signs` — uint8 [$n, d/8$]
- `{prefix}.norms` — float32 [$n$]
- `{prefix}.residual_norms` — float32 [$n$]
- `{prefix}.pi` — float32 [$d, d$] rotation matrix

### 1.2 Decode Pipeline (OpenCL Kernel)

The inverse kernel `turboquant_inverse_q4.opencl` reconstructs float32 vectors:

```
For each row i, element j:
  idx = mse_indices[i, j/2] >> (4 * (j%2)) & 0xF   (nibble unpack)
  centroid_val = centroids[idx]                      (Lloyd-Max lookup)
  sign_bit = (qjl_signs[i, j/8] >> (j%8)) & 1       (QJL sign)
  correction = qjl_scale * residual_norms[i] * (2*sign_bit - 1)
  rotated = centroid_val * norms[i] + correction
  output[i,j] = sum_k(rotated * pi_inv[j,k])         (inverse rotation)
```

### 1.3 Byte Accounting (per shardlet, 256 rows × 128 dim)

| Tensor | Dtype | Shape | Bytes |
|--------|-------|-------|-------|
| mse_indices | uint8 | [256, 64] | 16,384 |
| qjl_signs | uint8 | [256, 16] | 4,096 |
| norms | float32 | [256] | 1,024 |
| residual_norms | float32 | [256] | 1,024 |
| pi (rotation) | float32 | [128, 128] | 65,536 |
| **Stored total** | | | **88,064** |
| centroids (generated) | float32 | [8] | 32 |
| qjl_matrix (generated) | float32 | [128, 128] | 65,536 |
| **Decoded output** | float32 | [256, 128] | **131,072** |

Compression ratio (stored/decoded): $88064 / 131072 = 0.672$ (32.8% reduction).
Note: the rotation matrix $\Pi$ is amortized across all shardlets sharing the same
seed, so effective per-row storage is $(16384 + 4096 + 1024 + 1024) / 256 = 88$
bytes/row vs $128 \times 4 = 512$ bytes/row original = **5.8× compression**.

## 2. Prior Guidance: Theomatica + TPH Router

### 2.1 Theomatica_49GB Rotation Bases

The Theomatica_49GB model provides a set of rotation matrices derived from
structure-aware decomposition of model weight families. These serve as an
alternative to random rotation in Step 1:

- `flavor_best`: The rotation basis selected by prior shootout (experiment 0018)
  as the best-performing prior for each tensor family
- Used as $\Pi$ in place of random orthogonal matrix

### 2.2 TPH 202-Clone Router

202 TPH (Transformerless Predicted Hierarchy) model clones emerged incidentally
during quantization experimentation (see ray-quimb method). They function as a
prior-selection router:

- Each clone encodes a different quantization configuration
- The router selects which Theomatica rotation basis to apply per tensor slice
- Selection criterion: minimum reconstruction error on held-out rows

### 2.3 Controls

Every guided quantization result is compared against:

1. **Direct q4** — same Lloyd-Max + QJL pipeline with random rotation (no prior)
2. **Random basis** — same pipeline with a different random orthogonal matrix
3. **Identity** — no rotation applied

## 3. Validation Pipeline

### 3.1 Decode Correctness

The OpenCL kernel output is validated against the canonical PyTorch
`TurboQuantProd.dequantize()` reference:

```
max_abs_diff:  2.608e-08
mean_abs_diff: 1.774e-09
rms_diff:      2.530e-09
allclose(atol=1e-5): True
```

### 3.2 Makefile Verification Steps

| Step | What It Checks |
|------|----------------|
| validate | JSON schema + Python syntax of all package files |
| sidecar | OpenCL sidecar kernel compilation and buffer allocation |
| descriptor | Tensor shape/dtype/hash match against manifest |
| c-smoke | C-level GGML API smoke test |
| loader-smoke | GGML loader surface availability |
| framework | Full runner_framework.py end-to-end |
| release-check | Release lock integrity and version match |

### 3.3 Held-Out Metrics (from experiment series 0005–0018)

| Metric | What It Measures |
|--------|------------------|
| relative_error | $\|W - \hat{W}\| / \|W\|$ per tensor |
| amplitude_fidelity | $|\langle w, \hat{w} \rangle|^2$ (quantum-style) |
| phase_fidelity | Phase agreement after complex encoding |
| tph_729_fidelity | 729-state TPH encoder agreement |
| allclose_1e_5 | Boolean pass/fail at $10^{-5}$ tolerance |

## 4. Ray-Quimb Method

A ray-quimb virtual quantum method originated in vessel-production. It
demonstrated that the 202 TPH clones emerged from held-out bit changes during
quantization experimentation — not from deliberate design. The clones are
incidental prototypes from a test run accumulation.

The full development provenance lives in private lab history and is not part of
this public runner package.

## 5. GGML Tensor Materialization

Decoded float32 vectors are materialized into GGML tensor buffers via
`ggml_tensor_buffer_shim.py`, bridging the gap between the TurboQuant decode
output and llama.cpp's weight-loading interface:

```
packed shardlet → OpenCL decode → float32 [rows, 128] → ggml_tensor → llama.cpp
```

**Current status**: Decode-and-verify works. Named-tensor routing into llama.cpp's
actual inference weight loader is the next engineering step.

## 6. Claim Boundary

This package proves:
- One real q4 shardlet can be packed, stored, decoded on AMD GPU, and validated
- The decode path matches the canonical Torch reference at float32 precision
- The GGML tensor buffer shim can materialize decoded vectors

This package does **not** prove:
- Full 675B model inference through this pipeline
- Compression superiority over GPTQ, AWQ, or GGUF baselines
- Perplexity, LongBench, needle-in-haystack, or logit-KL quality metrics
- Native GGML weight loading for inference
