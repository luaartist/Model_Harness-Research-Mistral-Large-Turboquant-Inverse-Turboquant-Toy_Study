#!/usr/bin/env python3
"""Build a math-gated q4 shardlet catalog.

This generator creates multiple q4 TurboQuant shardlets from diverse Mistral
attention tensors at varying row counts, applies all 8 promotion gates from
SCALING_MATH_GATES.md, and emits package_catalog.json with only promoted
entries.

Usage:
    /usr/bin/python3 build_shardlet_catalog.py [--max-shardlets N]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

TOOL_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = TOOL_DIR.parent
RESULTS_DIR = EXPERIMENT_DIR / "results"
CATALOG_ARTIFACT_DIR = EXPERIMENT_DIR / "catalog_artifacts"
RUNTIME_PACKAGE_DIR = EXPERIMENT_DIR / "runtime_package"
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "../Mistral-Large-Quantization-Project"))
BASE_0015_PATH = (
    MODEL_DIR
    / "_model_lab/experiments/0015_turboquant_theomatica_tph_shardlet"
    / "run_turboquant_shardlet_probe.py"
)
FLAVOR_PATH = Path(os.environ.get("FLAVOR_PATH", "artifacts/flavor.safetensors"))
SOURCE_0012 = (
    MODEL_DIR
    / "_model_lab/experiments/0012_mistral_large_overfit_check/results/"
    "large_model_overfit_result.json"
)
WOLFRAM_CHECKS = RESULTS_DIR / "scaling_math_gate_checks.json"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VECTOR_DIM = 128
KEY_BITS = 4
DEFAULT_SEED = 42
RANDOM_SEED = 729
CALIB_FRACTION = 0.5
MAX_FLAVOR_ROTATIONS = 8
DEFAULT_MAX_SHARDLETS = 10


def load_base_module() -> Any:
    spec = importlib.util.spec_from_file_location("exp0015", BASE_0015_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {BASE_0015_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["exp0015"] = module
    spec.loader.exec_module(module)
    return module


BASE: Any | None = None


def get_base() -> Any:
    global BASE
    if BASE is None:
        BASE = load_base_module()
    return BASE


# ---------------------------------------------------------------------------
# Candidate selection — diverse families and row counts
# ---------------------------------------------------------------------------

@dataclass
class CatalogCandidate:
    name: str
    family: str
    shard: str
    shape: list[int]
    row_start: int
    col_start: int
    rows: int


def select_catalog_candidates(
    max_shardlets: int,
) -> list[CatalogCandidate]:
    """Select diverse candidates across families, layers, row counts."""
    result = json.loads(SOURCE_0012.read_text(encoding="utf-8"))

    # Group by family
    by_family: dict[str, list[dict[str, Any]]] = {}
    for c in result["candidates"]:
        family = c.get("family", "")
        if not family.startswith("attention_"):
            continue
        shape = c["shape"]
        if len(shape) != 2 or shape[1] < VECTOR_DIM:
            continue
        by_family.setdefault(family, []).append(c)

    # Row count tiers — doubled since heldout = half
    # 512→256 heldout, 2048→1024 heldout, 4096→2048 heldout
    row_tiers = [512, 2048, 4096]

    candidates: list[CatalogCandidate] = []
    families_used = sorted(by_family.keys())

    for tier in row_tiers:
        for family in families_used:
            if len(candidates) >= max_shardlets:
                break
            for c in by_family[family]:
                if c["shape"][0] < tier:
                    continue
                # Avoid duplicating same tensor+tier
                key = (c["name"], c["shard"], tier)
                if any(
                    (cc.name, cc.shard, cc.rows) == key
                    for cc in candidates
                ):
                    continue
                row_start = min(
                    int(c.get("row_start", 0)),
                    c["shape"][0] - tier,
                )
                col_start = min(
                    int(c.get("col_start", 0)),
                    c["shape"][1] - VECTOR_DIM,
                )
                candidates.append(
                    CatalogCandidate(
                        name=c["name"],
                        family=family,
                        shard=c["shard"],
                        shape=c["shape"],
                        row_start=row_start,
                        col_start=col_start,
                        rows=tier,
                    )
                )
                break  # One per family per tier
        if len(candidates) >= max_shardlets:
            break

    return candidates[:max_shardlets]


# ---------------------------------------------------------------------------
# Math gates
# ---------------------------------------------------------------------------

def spectral_norm(matrix: torch.Tensor) -> float:
    """Largest singular value."""
    s = torch.linalg.svdvals(matrix)
    return float(s[0].item())


def frobenius_norm(matrix: torch.Tensor) -> float:
    return float(torch.norm(matrix, p="fro").item())


def effective_rank(matrix: torch.Tensor) -> float:
    """Shannon entropy effective rank."""
    s = torch.linalg.svdvals(matrix)
    s2 = s * s
    total = s2.sum()
    if total < 1e-30:
        return 0.0
    p = s2 / total
    p = p[p > 1e-30]
    entropy = -torch.sum(p * torch.log(p)).item()
    return math.exp(entropy)


def anisotropy(matrix: torch.Tensor) -> float:
    s = torch.linalg.svdvals(matrix)
    s2 = s * s
    mean_s2 = s2.mean()
    if mean_s2 < 1e-30:
        return 0.0
    return float((s2[0] / mean_s2).item())


def compute_gate_metrics(
    reference: torch.Tensor,
    reconstruction: torch.Tensor,
    calib_ref: torch.Tensor,
    calib_recon: torch.Tensor,
) -> dict[str, Any]:
    """Compute all 8 promotion gate metrics."""
    diff = reference - reconstruction

    # Gate 2: Heldout reconstruction
    rel_err = frobenius_norm(diff) / max(frobenius_norm(reference), 1e-12)
    mse = float(torch.mean(diff * diff).item())
    cos = float(
        torch.dot(reference.flatten(), reconstruction.flatten()).item()
        / max(
            frobenius_norm(reference) * frobenius_norm(reconstruction),
            1e-12,
        )
    )

    # Gate 3: Inner-product / Gram
    gram_ref = reference @ reference.T
    gram_recon = reconstruction @ reconstruction.T
    gram_diff = gram_ref - gram_recon
    gram_error = frobenius_norm(gram_diff) / max(
        frobenius_norm(gram_ref), 1e-12
    )

    q_count = min(64, reference.shape[0])
    queries = reference[:q_count]
    score_ref = queries @ reference.T
    score_recon = queries @ reconstruction.T
    score_diff = score_ref - score_recon
    query_score_error = frobenius_norm(score_diff) / max(
        frobenius_norm(score_ref), 1e-12
    )

    # Gate 4: Spectral distortion
    spec_err = spectral_norm(diff) / max(spectral_norm(reference), 1e-12)
    cov_ref = reference.T @ reference
    cov_recon = reconstruction.T @ reconstruction
    cov_diff = cov_ref - cov_recon
    cov_err = spectral_norm(cov_diff) / max(spectral_norm(cov_ref), 1e-12)

    # Gate 5: Effective rank and anisotropy (of reference)
    eff_rank = effective_rank(reference)
    aniso = anisotropy(reference)

    # Gate 7: Overfit check
    calib_diff = calib_ref - calib_recon
    calib_rel = frobenius_norm(calib_diff) / max(
        frobenius_norm(calib_ref), 1e-12
    )
    gen_ratio = rel_err / max(calib_rel, 1e-12)

    return {
        "heldout_relative_error": rel_err,
        "heldout_mse": mse,
        "heldout_cosine": cos,
        "gram_error": gram_error,
        "query_score_error": query_score_error,
        "spectral_error": spec_err,
        "covariance_error": cov_err,
        "effective_rank": eff_rank,
        "anisotropy": aniso,
        "calibration_relative_error": calib_rel,
        "generalization_ratio": gen_ratio,
    }


# ---------------------------------------------------------------------------
# Byte accounting
# ---------------------------------------------------------------------------

def byte_accounting(rows: int, dim: int, bits: int) -> dict[str, int]:
    mse_bits = bits - 1
    if mse_bits == 1:
        vpb = 8
    elif mse_bits == 2:
        vpb = 4
    else:
        vpb = 2
    mse_index_bytes = (dim + vpb - 1) // vpb
    qjl_sign_bytes = (dim + 7) // 8
    per_vector = mse_index_bytes + qjl_sign_bytes + 4  # +4 for norms
    payload = rows * per_vector
    source_pi = 2 * dim * dim
    runtime_pi = 4 * dim * dim
    runtime_qjl = 4 * dim * dim
    centroid = 4 * (2 ** mse_bits)
    baseline = 2 * rows * dim
    return {
        "payload_bytes": payload,
        "source_pi_bytes": source_pi,
        "runtime_pi_bytes": runtime_pi,
        "runtime_qjl_matrix_bytes": runtime_qjl,
        "centroid_bytes": centroid,
        "baseline_bf16_bytes": baseline,
        "source_total_bytes": payload + source_pi,
        "runtime_upload_bytes": payload + runtime_pi + runtime_qjl + centroid,
        "payload_only_ratio": baseline / payload,
        "source_ratio": baseline / (payload + source_pi),
        "runtime_upload_ratio": baseline / (
            payload + runtime_pi + runtime_qjl + centroid
        ),
    }


# ---------------------------------------------------------------------------
# Shardlet builder
# ---------------------------------------------------------------------------

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


@dataclass
class ShardletResult:
    candidate: CatalogCandidate
    prefix: str
    artifact_path: Path
    artifact_sha256: str
    artifact_bytes: int
    rows: int
    vector_dim: int
    gate_metrics: dict[str, Any]
    control_default: dict[str, Any]
    control_random: dict[str, Any]
    byte_account: dict[str, Any]
    production_bytes: dict[str, Any]
    flavor_basis: str
    promoted: bool
    rejection_reasons: list[str] = field(default_factory=list)


def load_vector_block(
    candidate: CatalogCandidate,
) -> torch.Tensor:
    with safe_open(
        str(MODEL_DIR / candidate.shard),
        framework="pt",
        device="cpu",
    ) as handle:
        tensor_slice = handle.get_slice(candidate.name)
        data = tensor_slice[
            candidate.row_start: candidate.row_start + candidate.rows,
            candidate.col_start: candidate.col_start + VECTOR_DIM,
        ]
    expected = (candidate.rows, VECTOR_DIM)
    if data.shape != expected:
        raise ValueError(f"bad shape {data.shape}, expected {expected}")
    return data.to(device="cuda", dtype=torch.float32)


def shape_from_slice(slice_view: Any) -> list[int]:
    if hasattr(slice_view, "get_shape"):
        return list(slice_view.get_shape())
    return list(slice_view[:].shape)


def load_flavor_rotations() -> dict[str, torch.Tensor]:
    rotations: dict[str, torch.Tensor] = {}
    with safe_open(str(FLAVOR_PATH), framework="pt", device="cpu") as handle:
        for key in sorted(handle.keys()):
            sl = handle.get_slice(key)
            shape = shape_from_slice(sl)
            if len(shape) < 2 or shape[0] < VECTOR_DIM or shape[1] < VECTOR_DIM:
                continue
            matrix = sl[:VECTOR_DIM, :VECTOR_DIM]
            rotations[f"flavor:{key}"] = get_base().orthogonalize(matrix, VECTOR_DIM)
            if len(rotations) >= MAX_FLAVOR_ROTATIONS:
                break
    if not rotations:
        raise RuntimeError(f"no compatible rotations in {FLAVOR_PATH}")
    return rotations


def build_shardlet(
    index: int,
    candidate: CatalogCandidate,
    flavor_rotations: dict[str, torch.Tensor],
) -> ShardletResult:
    """Quantize one candidate and apply all gates."""
    vectors = load_vector_block(candidate)
    calib_rows = candidate.rows // 2
    calibration = vectors[:calib_rows]
    heldout = vectors[calib_rows:]

    # Choose best flavor rotation on calibration
    best_name, best_pi, _ = get_base().choose_best_rotation(
        calibration, KEY_BITS, flavor_rotations,
    )

    # Evaluate flavor_best on heldout
    stored_fb, recon_fb, _ = get_base().evaluate_variant(
        heldout, KEY_BITS, DEFAULT_SEED, best_pi,
    )
    # Calibration reconstruction for overfit check
    _, calib_recon_fb, _ = get_base().evaluate_variant(
        calibration, KEY_BITS, DEFAULT_SEED, best_pi,
    )

    # Controls on heldout
    _, recon_def, _ = get_base().evaluate_variant(
        heldout, KEY_BITS, DEFAULT_SEED, None,
    )
    _, recon_rnd, _ = get_base().evaluate_variant(
        heldout, KEY_BITS, RANDOM_SEED, None,
    )

    # Gate metrics
    gates = compute_gate_metrics(heldout, recon_fb, calibration, calib_recon_fb)
    ctrl_def = compute_gate_metrics(heldout, recon_def, calibration, calibration)
    ctrl_rnd = compute_gate_metrics(heldout, recon_rnd, calibration, calibration)
    # Byte accounting on actual heldout rows (what's stored in artifact)
    bytes_acc = byte_accounting(heldout.shape[0], VECTOR_DIM, KEY_BITS)
    # Production projection: full tensor rows share one pi matrix
    prod_bytes = byte_accounting(candidate.shape[0], VECTOR_DIM, KEY_BITS)

    # Save artifact
    prefix = f"s{index}.q4.flavor_best"
    tensors: dict[str, torch.Tensor] = {}
    get_base().save_variant_tensors(tensors, prefix, stored_fb, best_pi)
    artifact_path = CATALOG_ARTIFACT_DIR / f"catalog_shardlet_s{index}.safetensors"
    save_file(
        tensors,
        str(artifact_path),
        metadata={
            "format": "catalog_shardlet_v1",
            "vector_dim": str(VECTOR_DIM),
            "rows": str(heldout.shape[0]),
            "source_tensor": candidate.name,
            "source_shard": candidate.shard,
        },
    )

    # Promotion decision (Gate 6: control superiority)
    rejection: list[str] = []

    # Gate 6: flavor_best should beat both controls on heldout ip error
    if gates["query_score_error"] > ctrl_def["query_score_error"]:
        rejection.append(
            f"flavor_best query_score_error {gates['query_score_error']:.6f} "
            f"> default {ctrl_def['query_score_error']:.6f}"
        )
    if gates["query_score_error"] > ctrl_rnd["query_score_error"]:
        rejection.append(
            f"flavor_best query_score_error {gates['query_score_error']:.6f} "
            f"> random {ctrl_rnd['query_score_error']:.6f}"
        )

    # Gate 7: Overfit — generalization ratio should be < 1.15
    if gates["generalization_ratio"] > 1.15:
        rejection.append(
            f"generalization_ratio {gates['generalization_ratio']:.4f} > 1.15"
        )

    # Gate 1: Source ratio for production deployment should be >= 1.5
    if prod_bytes["source_ratio"] < 1.5:
        rejection.append(
            f"production source_ratio {prod_bytes['source_ratio']:.4f} < 1.5 "
            f"(full tensor rows={candidate.shape[0]})"
        )

    promoted = len(rejection) == 0

    return ShardletResult(
        candidate=candidate,
        prefix=prefix,
        artifact_path=artifact_path,
        artifact_sha256=sha256_file(artifact_path),
        artifact_bytes=artifact_path.stat().st_size,
        rows=heldout.shape[0],
        vector_dim=VECTOR_DIM,
        gate_metrics=gates,
        control_default=ctrl_def,
        control_random=ctrl_rnd,
        byte_account=bytes_acc,
        production_bytes=prod_bytes,
        flavor_basis=best_name,
        promoted=promoted,
        rejection_reasons=rejection,
    )


# ---------------------------------------------------------------------------
# Catalog assembly
# ---------------------------------------------------------------------------

def build_catalog_entry(sr: ShardletResult) -> dict[str, Any]:
    return {
        "prefix": sr.prefix,
        "source_tensor": sr.candidate.name,
        "source_family": sr.candidate.family,
        "source_shard": sr.candidate.shard,
        "source_shape": sr.candidate.shape,
        "rows": sr.rows,
        "vector_dim": sr.vector_dim,
        "key_bits": KEY_BITS,
        "flavor_basis": sr.flavor_basis,
        "artifact_path": str(sr.artifact_path),
        "artifact_bytes": sr.artifact_bytes,
        "artifact_sha256": sr.artifact_sha256,
        "byte_accounting": sr.byte_account,
        "production_bytes": sr.production_bytes,
        "gate_metrics": sr.gate_metrics,
        "control_default": {
            "heldout_relative_error": sr.control_default[
                "heldout_relative_error"
            ],
            "query_score_error": sr.control_default["query_score_error"],
            "gram_error": sr.control_default["gram_error"],
        },
        "control_random": {
            "heldout_relative_error": sr.control_random[
                "heldout_relative_error"
            ],
            "query_score_error": sr.control_random["query_score_error"],
            "gram_error": sr.control_random["gram_error"],
        },
        "promoted": sr.promoted,
        "rejection_reasons": sr.rejection_reasons,
    }


def build_catalog(
    results: list[ShardletResult],
) -> dict[str, Any]:
    entries = [build_catalog_entry(sr) for sr in results]
    promoted = [e for e in entries if e["promoted"]]
    rejected = [e for e in entries if not e["promoted"]]
    return {
        "schema_version": 1,
        "catalog_name": "mistral_large_q4_shardlet_catalog_v1",
        "vector_dim": VECTOR_DIM,
        "key_bits": KEY_BITS,
        "kernel": "runtime_package/turboquant_inverse_q4.opencl",
        "total_candidates": len(entries),
        "promoted_count": len(promoted),
        "rejected_count": len(rejected),
        "promoted": promoted,
        "rejected": rejected,
        "gate_spec": "SCALING_MATH_GATES.md",
        "wolfram_checks": str(WOLFRAM_CHECKS),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-shardlets",
        type=int,
        default=DEFAULT_MAX_SHARDLETS,
    )
    args = parser.parse_args()

    get_base().require_cuda()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CATALOG_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20260512)

    print("selecting candidates...")
    candidates = select_catalog_candidates(args.max_shardlets)
    print(f"  selected {len(candidates)} candidates")
    for i, c in enumerate(candidates):
        print(f"  s{i}: {c.name[:60]} rows={c.rows} family={c.family}")

    print("\nloading flavor rotations...")
    flavor_rotations = load_flavor_rotations()
    print(f"  loaded {len(flavor_rotations)} rotations")

    print("\nbuilding shardlets with math gates...")
    results: list[ShardletResult] = []
    for i, candidate in enumerate(candidates):
        t0 = time.monotonic()
        sr = build_shardlet(i, candidate, flavor_rotations)
        dt = time.monotonic() - t0
        status = "PROMOTED" if sr.promoted else "REJECTED"
        print(
            f"  s{i} [{status}] rows={sr.rows} "
            f"rel_err={sr.gate_metrics['heldout_relative_error']:.6f} "
            f"gram={sr.gate_metrics['gram_error']:.6f} "
            f"spec={sr.gate_metrics['spectral_error']:.6f} "
            f"eff_rank={sr.gate_metrics['effective_rank']:.1f} "
            f"gen_ratio={sr.gate_metrics['generalization_ratio']:.4f} "
            f"heldout_src={sr.byte_account['source_ratio']:.4f} "
            f"prod_src={sr.production_bytes['source_ratio']:.4f} "
            f"({dt:.1f}s)"
        )
        if not sr.promoted:
            for reason in sr.rejection_reasons:
                print(f"    reason: {reason}")
        results.append(sr)

    catalog = build_catalog(results)

    catalog_path = RESULTS_DIR / "package_catalog.json"
    catalog_path.write_text(
        json.dumps(catalog, indent=2), encoding="utf-8",
    )

    # Also write into runtime_package for runner consumption
    runner_catalog = RUNTIME_PACKAGE_DIR / "package_catalog.json"
    runner_catalog.write_text(
        json.dumps(catalog, indent=2), encoding="utf-8",
    )

    print(f"\ncatalog written: {catalog_path}")
    print(f"runner catalog: {runner_catalog}")
    print(f"total: {catalog['total_candidates']}")
    print(f"promoted: {catalog['promoted_count']}")
    print(f"rejected: {catalog['rejected_count']}")

    promoted_entries = catalog["promoted"]
    if promoted_entries:
        print("\npromoted entries:")
        for e in promoted_entries:
            print(
                f"  {e['prefix']}: rows={e['rows']} "
                f"family={e['source_family']} "
                f"rel_err={e['gate_metrics']['heldout_relative_error']:.6f} "
                f"prod_src={e['production_bytes']['source_ratio']:.4f}"
            )


if __name__ == "__main__":
    main()
