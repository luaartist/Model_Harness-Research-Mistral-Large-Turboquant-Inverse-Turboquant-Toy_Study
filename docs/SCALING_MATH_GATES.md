# Math Gates For Catalog Scaling

The next scaling step should not only emit more shardlets. It should promote a
shardlet into the catalog only after it survives mathematical checks that are
harder than the v0.1 single-shardlet decode proof.

## Scope

Target the same tensor family and decode contract first:

```text
variant: c*.q4.flavor_best
vector_dim: 128
kernel: runtime_package/turboquant_inverse_q4.opencl
artifact format: prior_shootout_v1-compatible safetensors tensors
runner output: row-major float32 [rows, 128]
```

The catalog should contain 5 to 10 promoted shardlets plus their controls. A
candidate can be recorded as a negative result, but it should not be promoted as
a runner catalog entry unless the gates below pass.

## Required Gates

1. **Byte Accounting Gate**

   Record persistent artifact bytes and runtime upload bytes separately.

   ```text
   payload_bytes(r, d, b) = r * (mse_index_bytes(d, b) + qjl_sign_bytes(d) + 4)
   mse_index_bytes(d, 4) = ceil(d / 2)
   qjl_sign_bytes(d) = ceil(d / 8)
   source_pi_bytes(d) = 2 * d * d
   runtime_pi_bytes(d) = 4 * d * d
   runtime_qjl_matrix_bytes(d) = 4 * d * d
   baseline_bf16_bytes(r, d) = 2 * r * d
   ```

   For `d=128`, `b=4`, the payload is 84 bytes per vector and the asymptotic
   payload-only compression ratio is about 3.0476x. Source artifact ratios are
   much lower for small row counts because the `pi` matrix is stored once per
   shardlet. This is why the next catalog must include larger row-count tests,
   not only more 256-row slices.

2. **Heldout Reconstruction Gate**

   Keep the existing calibration/heldout split. The catalog decision must use
   heldout rows only.

   ```text
   relative_error = ||X - X_hat||_F / ||X||_F
   mse = mean((X - X_hat)^2)
   cosine = dot(vec(X), vec(X_hat)) / (||X||_F ||X_hat||_F)
   ```

3. **Inner-Product Gate**

   Preserve the metric that matters for attention-like use:

   ```text
   gram_error = ||X X^T - X_hat X_hat^T||_F / ||X X^T||_F
   query_score_error = ||Q X^T - Q X_hat^T||_F / ||Q X^T||_F
   ```

   `query_score_error` should continue to use the same query convention as the
   0015/0018 harness so results are comparable.

4. **Spectral Distortion Gate**

   Add a stricter matrix-level check:

   ```text
   spectral_error = ||X - X_hat||_2 / ||X||_2
   covariance_error = ||X^T X - X_hat^T X_hat||_2 / ||X^T X||_2
   ```

   This prevents a shardlet from passing by average error while damaging a
   dominant direction.

5. **Effective-Rank And Anisotropy Gate**

   Record why the candidate is hard:

   ```text
   p_i = sigma_i^2 / sum_j sigma_j^2
   effective_rank = exp(-sum_i p_i log(p_i))
   anisotropy = sigma_1^2 / mean_i(sigma_i^2)
   ```

   The promoted catalog should cover diverse spectra, not 10 nearly identical
   easy windows.

6. **Control Superiority Gate**

   Promote `flavor_best` only when it is heldout-competitive against both
   controls:

   ```text
   tq_default
   tq_random_seed729
   ```

   The primary comparison is heldout inner-product error. Gram and spectral
   errors are tie-breakers. A failed `flavor_best` row should be kept as a
   negative result, not hidden.

7. **Overfit Gate**

   Record the calibration-to-heldout gap:

   ```text
   generalization_ratio = heldout_loss / max(calibration_loss, epsilon)
   ```

   A rotation that wins calibration but collapses on heldout is rejected even if
   its raw heldout number is superficially tolerable.

8. **Runtime Identity Gate**

   Every promoted shardlet must decode through the same OpenCL kernel and match
   the Torch reference within the existing float32 tolerance. The catalog gate
   is not allowed to introduce a per-shardlet kernel.

## Wolfram Role

Use Wolfram Engine for the parts that should be exact or symbolically checked:

```text
byte-accounting formulas
row thresholds for pi-overhead amortization
matrix norm identities on small deterministic matrices
spectral/effective-rank formulas
JSON emission of expected byte tables
```

The first check script is:

```bash
wolframscript -script scaling_math_gate_checks.wl
```

It writes:

```text
results/scaling_math_gate_checks.json
```

## Grok Role

Use Grok as a skeptical reviewer, not as an authority for hard thresholds. Its
recommendations are useful for surfacing missing gates, but thresholds should be
derived from existing 0015/0018 distributions, controls, and byte accounting.

## Catalog Promotion Record

Each promoted catalog entry should contain:

```text
artifact path and SHA-256
prefix
rows and vector_dim
payload/source/runtime byte accounting
heldout reconstruction metrics
heldout Gram/query-score metrics
spectral/covariance metrics
effective rank and anisotropy
control comparison
generalization ratio
sidecar output SHA-256
descriptor path and SHA-256
C runner audit path and SHA-256
kernel SHA-256
promotion decision
```

Exit criterion: `package_catalog.json` contains 5 to 10 promoted shardlets,
all sharing one decode kernel, and `runner_framework.py --catalog
package_catalog.json` verifies every entry.
