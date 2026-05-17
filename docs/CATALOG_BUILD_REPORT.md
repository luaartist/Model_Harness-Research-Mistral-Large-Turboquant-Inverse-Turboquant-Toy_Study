# Catalog Build Report — v1 Math-Gated Shardlet Catalog

**Date:** 2025-05-12  
**Generator:** `build_shardlet_catalog.py`  
**Wolfram audit:** `results/wolfram_catalog_byte_audit.json`  
**Grok review:** `results/grok_catalog_review.json`  
**Catalog:** `results/package_catalog.json` → `runtime_package/package_catalog.json`

---

## Summary

| Metric | Value |
|--------|-------|
| Total candidates evaluated | 10 |
| Promoted | 4 |
| Rejected | 6 |
| Families covered | 2 (attention_wo, attention_wq_a) |
| Row tiers covered | 3 (256, 1024, 2048 heldout) |
| Shared kernel | turboquant_inverse_q4.opencl |

## Promoted Entries

| Prefix | Family | Heldout Rows | Rel Error | Gram Error | Spectral | Gen Ratio | Prod Source Ratio |
|--------|--------|-------------|-----------|------------|----------|-----------|-------------------|
| s2.q4.flavor_best | attention_wo | 256 | 0.2280 | 0.1312 | 0.1315 | 0.9889 | 2.89x |
| s3.q4.flavor_best | attention_wq_a | 256 | 0.2300 | 0.1944 | 0.2350 | 1.0130 | 2.43x |
| s6.q4.flavor_best | attention_wo | 1024 | 0.2299 | 0.1314 | 0.1256 | 1.0036 | 2.89x |
| s9.q4.flavor_best | attention_wo | 2048 | 0.2309 | 0.1326 | 0.1230 | 1.0061 | 2.89x |

## 8-Gate Results

1. **Byte accounting** — Wolfram-verified. Production source ratios 2.43–2.89x. All > 1.5x threshold.
2. **Heldout reconstruction** — Relative error 0.228–0.231. Stable across row counts.
3. **Inner-product / Gram** — Gram error 0.131–0.194. Query score error beats controls.
4. **Spectral distortion** — 0.123–0.235. Consistent with q4 compression.
5. **Effective rank / Anisotropy** — Rank 37–58. Data is well-distributed.
6. **Control superiority** — All promoted entries beat random AND default controls on query_score_error.
7. **Overfit check** — Gen ratios 0.989–1.013. No overfitting detected.
8. **Runtime identity** — Deferred to Phase 1 (C http_bridge OpenCL decode round-trip).

## Wolfram Verification

All 4 promoted entries verified via `wolframscript`:
- per_vector_payload = 84 bytes
- source_pi_overhead = 32,768 bytes (f16 rotation matrix)
- Production source ratios match Python to 12 decimal places

## Grok Skeptical Review (grok-4-1-fast-reasoning)

**Disposition: PROCEED_WITH_CAVEATS**

| Finding | Risk | Action |
|---------|------|--------|
| Gen ratio valid but split lacks diversity stats | Low | Acknowledged — same-shard split is standard for q4 probes |
| 3/4 entries from attention_wo (family bias) | Medium | Diversify in v2 catalog when more families pass flavor rotation gate |
| Gram error 13–19% marginal | Medium | Monitor downstream perplexity when E2E decode available |
| Single kernel compatibility | Low | Kernel is per-vector, row count irrelevant — risk overstated |
| Missing E2E perplexity gate | High | Deferred to Phase 1 C http_bridge |

## Rejection Analysis

6 candidates rejected, all for the same reason: **flavor_best query_score_error > control**.
This means the flavor rotation did not improve inner-product preservation for those tensors.
The families affected (wkv_a_with_mqa, wkv_b, wq_b) may need different rotation strategies.

---

## Next Steps

### Immediate (Phase 0L Complete → Phase 1 Entry)

1. **C http_bridge backend** — Build the C file-backed HTTP bridge that accepts
   `package_catalog.json`, loads each shardlet artifact, compiles the OpenCL kernel,
   and runs decode/verify cycles. This enables Gate 8 (runtime identity) and unlocks
   the E2E perplexity gate Grok flagged.

2. **Runner catalog mode** — Extend `runner_framework.py --catalog package_catalog.json`
   to iterate promoted entries, run sidecar decode on each, and verify hash stability.

3. **Diversify catalog v2** — Investigate why wkv_b and wq_b fail the control superiority
   gate. Options:
   - More flavor rotation candidates (currently limited to 8)
   - Different calibration split strategies
   - Family-specific rotation search (e.g., Theomatica rotations for wkv_b)

### Medium-term (Phase 1 → Phase 2)

4. **Multi-shard coverage** — Current catalog uses only layer 0 tensors. Extend to
   layers 1–60 to verify cross-layer consistency.

5. **E2E perplexity delta** — Once C bridge enables full-layer decode, measure
   perplexity impact of q4 quantization vs bf16 baseline on a held-out token set.

6. **Production packaging** — Move from proof-of-concept catalog to production
   `package_catalog.json` with all 61 layers × relevant attention families.
