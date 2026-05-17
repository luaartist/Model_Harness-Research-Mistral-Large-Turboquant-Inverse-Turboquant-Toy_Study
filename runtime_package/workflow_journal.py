#!/usr/bin/env python3
"""Persistent workflow journal for the shardlet runtime package.

Tracks completed steps, their outputs, timing, and hashes so a new session
can read the journal and know exactly where the build stands without
re-scanning or re-running anything.

Usage:
    # Record a step completion
    python3 workflow_journal.py record \
        --step validate --status pass --note "17 hash records"

    # Show current state (what's done, what's stale, what's next)
    python3 workflow_journal.py status

    # Emit machine-readable state for an LLM or script
    python3 workflow_journal.py context

    # Mark a step stale (e.g., after editing source)
    python3 workflow_journal.py invalidate --step c-smoke

    # Run a Makefile target and journal the result automatically
    python3 workflow_journal.py run --target verify
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


PACKAGE_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = PACKAGE_DIR.parent
RESULTS_DIR = EXPERIMENT_DIR / "results"
JOURNAL_PATH = RESULTS_DIR / "workflow_journal.json"

# Ordered pipeline steps — each maps to a Makefile target or logical phase.
# Dependencies encode which earlier steps must be valid before this one runs.
PIPELINE = [
    {
        "step": "validate",
        "target": "validate",
        "description": "SHA-256 manifest check on all package files",
        "dependencies": [],
        "output_files": ["results/runtime_package_validation.json"],
    },
    {
        "step": "sidecar",
        "target": "sidecar",
        "description": "Python OpenCL sidecar decode (produces .f32 + metadata)",
        "dependencies": ["validate"],
        "output_files": [
            "results/ggml_opencl_sidecar_decode.json",
            "results/ggml_opencl_sidecar_decode.f32",
        ],
    },
    {
        "step": "descriptor",
        "target": "descriptor",
        "description": "GGML tensor buffer shim descriptor",
        "dependencies": ["sidecar"],
        "output_files": ["results/ggml_tensor_buffer_shim.json"],
    },
    {
        "step": "c-smoke",
        "target": "c-smoke",
        "description": "Build + run native C tensor shim smoke test",
        "dependencies": ["sidecar"],
        "output_files": ["results/private_runner_api_audit.json"],
    },
    {
        "step": "loader-smoke",
        "target": "loader-smoke",
        "description": "Build + run GGML loader adapter smoke test",
        "dependencies": ["sidecar"],
        "output_files": ["results/ggml_loader_surface_audit.json"],
    },
    {
        "step": "framework",
        "target": "framework",
        "description": "Python runner framework audit (orchestrates all sub-audits)",
        "dependencies": ["validate", "descriptor", "c-smoke", "loader-smoke"],
        "output_files": ["results/python_runner_framework_audit.json"],
    },
    {
        "step": "release-check",
        "target": "release-check",
        "description": "Verify release lock hashes match current artifacts",
        "dependencies": ["framework"],
        "output_files": [],
    },
    {
        "step": "catalog",
        "target": None,  # run via runner_framework.py --catalog
        "description": "Catalog mode: integrity + decode + identity for promoted entries",
        "dependencies": ["validate"],
        "output_files": ["results/catalog_run_report.json"],
    },
    {
        "step": "gguf-write",
        "target": "gguf-write",
        "description": "Write decoded f32 shardlet to named GGUF tensor file",
        "dependencies": ["sidecar"],
        "output_files": ["results/gguf_shardlet_writer.json"],
    },
]

STEP_NAMES = [s["step"] for s in PIPELINE]
STEP_MAP = {s["step"]: s for s in PIPELINE}


# ---------------------------------------------------------------------------
# Journal I/O
# ---------------------------------------------------------------------------


def load_journal() -> dict[str, Any]:
    """Load or create the workflow journal."""
    if JOURNAL_PATH.exists():
        data = json.loads(JOURNAL_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "entries" in data:
            return data
    return {
        "schema_version": 1,
        "experiment": "0021_baremetal_llamacpp_coupling",
        "created": _now_iso(),
        "entries": {},
        "log": [],
    }


def save_journal(journal: dict[str, Any]) -> None:
    """Write journal to disk."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    journal["updated"] = _now_iso()
    JOURNAL_PATH.write_text(
        json.dumps(journal, indent=2, sort_keys=False),
        encoding="utf-8",
    )


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(
        timespec="seconds"
    )


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Step state
# ---------------------------------------------------------------------------


def output_fingerprint(step_def: dict[str, Any]) -> dict[str, Any]:
    """Hash the output files of a step to detect staleness."""
    fingerprints: dict[str, Any] = {}
    for relpath in step_def["output_files"]:
        full = EXPERIMENT_DIR / relpath
        if full.exists():
            fingerprints[relpath] = {
                "bytes": full.stat().st_size,
                "sha256": _sha256_file(full),
                "mtime": full.stat().st_mtime,
            }
        else:
            fingerprints[relpath] = None
    return fingerprints


def is_stale(
    journal: dict[str, Any],
    step_name: str,
) -> tuple[bool, str]:
    """Check if a previously-recorded step is stale (outputs changed)."""
    entry = journal["entries"].get(step_name)
    if entry is None:
        return True, "never recorded"
    if entry.get("status") != "pass":
        return True, f"last status: {entry.get('status', 'unknown')}"
    step_def = STEP_MAP.get(step_name)
    if step_def is None:
        return True, "unknown step"
    # Check output file fingerprints
    recorded_fp = entry.get("output_fingerprints", {})
    current_fp = output_fingerprint(step_def)
    for relpath, cur in current_fp.items():
        rec = recorded_fp.get(relpath)
        if cur is None:
            return True, f"output missing: {relpath}"
        if rec is None:
            return True, f"output not previously recorded: {relpath}"
        if cur["sha256"] != rec["sha256"]:
            return True, f"output changed: {relpath}"
    # Check dependencies
    for dep in step_def["dependencies"]:
        dep_stale, dep_reason = is_stale(journal, dep)
        if dep_stale:
            return True, f"dependency {dep} stale: {dep_reason}"
    return False, "current"


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_record(args: argparse.Namespace) -> None:
    """Record a step completion."""
    if args.step not in STEP_MAP:
        print(f"error: unknown step '{args.step}'. valid: {STEP_NAMES}")
        raise SystemExit(1)
    journal = load_journal()
    step_def = STEP_MAP[args.step]
    fp = output_fingerprint(step_def)
    entry = {
        "status": args.status,
        "recorded_at": _now_iso(),
        "note": args.note or "",
        "output_fingerprints": fp,
        "duration_seconds": args.duration,
    }
    journal["entries"][args.step] = entry
    journal["log"].append({
        "action": "record",
        "step": args.step,
        "status": args.status,
        "at": _now_iso(),
        "note": args.note or "",
    })
    save_journal(journal)
    print(f"recorded: {args.step} = {args.status}")


def cmd_invalidate(args: argparse.Namespace) -> None:
    """Mark a step (and its dependents) stale."""
    journal = load_journal()
    invalidated: list[str] = []
    for step_def in PIPELINE:
        name = step_def["step"]
        if name == args.step or args.step in step_def["dependencies"]:
            if name in journal["entries"]:
                journal["entries"][name]["status"] = "stale"
                invalidated.append(name)
    journal["log"].append({
        "action": "invalidate",
        "step": args.step,
        "cascade": invalidated,
        "at": _now_iso(),
    })
    save_journal(journal)
    print(f"invalidated: {', '.join(invalidated) or 'nothing to invalidate'}")


def cmd_status(args: argparse.Namespace) -> None:
    """Print human-readable workflow state."""
    journal = load_journal()
    version_path = PACKAGE_DIR / "VERSION"
    version = version_path.read_text().strip() if version_path.exists() else "?"

    print(f"experiment: 0021_baremetal_llamacpp_coupling")
    print(f"version:    {version}")
    print(f"journal:    {JOURNAL_PATH}")
    print()

    max_name = max(len(s["step"]) for s in PIPELINE)
    for step_def in PIPELINE:
        name = step_def["step"]
        entry = journal["entries"].get(name)
        if entry is None:
            tag = "[ ]"
            detail = "not run"
        elif entry["status"] == "pass":
            stale, reason = is_stale(journal, name)
            if stale:
                tag = "[~]"
                detail = f"stale: {reason}"
            else:
                tag = "[x]"
                ts = entry.get("recorded_at", "")
                dur = entry.get("duration_seconds")
                detail = ts
                if dur is not None:
                    detail += f"  ({dur:.1f}s)"
                note = entry.get("note", "")
                if note:
                    detail += f"  {note}"
        elif entry["status"] == "stale":
            tag = "[~]"
            detail = "marked stale"
        else:
            tag = "[!]"
            detail = f"status={entry['status']}"
        print(f"  {tag} {name:<{max_name}}  {detail}")

    # Log tail
    log = journal.get("log", [])
    if log:
        print(f"\nlast 5 journal entries:")
        for entry in log[-5:]:
            action = entry.get("action", "?")
            step = entry.get("step", "?")
            at = entry.get("at", "?")
            note = entry.get("note", "")
            line = f"  {at}  {action} {step}"
            if note:
                line += f"  ({note})"
            print(line)


def cmd_context(args: argparse.Namespace) -> None:
    """Emit machine-readable context blob for LLM consumption."""
    journal = load_journal()
    version_path = PACKAGE_DIR / "VERSION"
    version = version_path.read_text().strip() if version_path.exists() else "?"

    # Git info
    git_head = ""
    git_dirty = False
    try:
        git_head = subprocess.run(
            ["git", "log", "--oneline", "-1"],
            cwd=str(EXPERIMENT_DIR),
            capture_output=True,
            text=True,
        ).stdout.strip()
        git_status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(EXPERIMENT_DIR),
            capture_output=True,
            text=True,
        ).stdout.strip()
        git_dirty = bool(git_status)
    except FileNotFoundError:
        pass

    steps_state: dict[str, Any] = {}
    for step_def in PIPELINE:
        name = step_def["step"]
        entry = journal["entries"].get(name)
        stale, reason = is_stale(journal, name)
        steps_state[name] = {
            "status": entry["status"] if entry else "not_run",
            "stale": stale,
            "stale_reason": reason if stale else None,
            "recorded_at": entry.get("recorded_at") if entry else None,
            "note": entry.get("note", "") if entry else "",
            "description": step_def["description"],
        }

    context = {
        "experiment": "0021_baremetal_llamacpp_coupling",
        "version": version,
        "git_head": git_head,
        "git_dirty": git_dirty,
        "journal_path": str(JOURNAL_PATH),
        "pipeline_steps": steps_state,
        "log_tail": journal.get("log", [])[-10:],
    }
    print(json.dumps(context, indent=2))


def cmd_run(args: argparse.Namespace) -> None:
    """Run a Makefile target and journal the result."""
    step_name = args.step or args.target
    if step_name not in STEP_MAP:
        print(f"error: unknown step '{step_name}'. valid: {STEP_NAMES}")
        raise SystemExit(1)
    step_def = STEP_MAP[step_name]
    target = step_def.get("target")
    if target is None:
        print(f"error: step '{step_name}' has no Makefile target (run manually)")
        raise SystemExit(1)

    # Check dependencies
    journal = load_journal()
    for dep in step_def["dependencies"]:
        stale, reason = is_stale(journal, dep)
        if stale:
            print(f"warning: dependency '{dep}' is stale: {reason}")

    print(f"running: make {target}")
    t0 = time.monotonic()
    result = subprocess.run(
        ["make", target],
        cwd=str(PACKAGE_DIR),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    duration = round(time.monotonic() - t0, 2)
    status = "pass" if result.returncode == 0 else "fail"

    # Capture tail of output for log
    output_tail = result.stdout[-2000:] if result.stdout else ""

    fp = output_fingerprint(step_def)
    journal = load_journal()  # re-read in case something changed
    journal["entries"][step_name] = {
        "status": status,
        "recorded_at": _now_iso(),
        "note": f"make {target} exit={result.returncode}",
        "output_fingerprints": fp,
        "duration_seconds": duration,
    }
    journal["log"].append({
        "action": "run",
        "step": step_name,
        "target": target,
        "status": status,
        "exit_code": result.returncode,
        "duration_seconds": duration,
        "output_tail": output_tail,
        "at": _now_iso(),
    })
    save_journal(journal)

    # Print output
    if result.stdout:
        print(result.stdout)
    print(f"\n--- journal: {step_name} = {status} ({duration}s) ---")
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def cmd_run_all(args: argparse.Namespace) -> None:
    """Run all pipeline steps in order, skipping those already current."""
    journal = load_journal()
    for step_def in PIPELINE:
        name = step_def["step"]
        target = step_def.get("target")
        if target is None:
            print(f"  skip {name} (no Makefile target)")
            continue
        stale, reason = is_stale(journal, name)
        if not stale:
            print(f"  skip {name} (current)")
            continue
        print(f"\n=== {name}: {step_def['description']} ===")
        # Simulate args for cmd_run
        run_args = argparse.Namespace(step=name, target=target)
        cmd_run(run_args)
        journal = load_journal()  # re-read after each step


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Persistent workflow journal for the shardlet runtime package.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # record
    rec = sub.add_parser("record", help="Record a step completion")
    rec.add_argument("--step", required=True, choices=STEP_NAMES)
    rec.add_argument("--status", required=True, choices=["pass", "fail", "skip"])
    rec.add_argument("--note", default="")
    rec.add_argument("--duration", type=float, default=None)

    # invalidate
    inv = sub.add_parser("invalidate", help="Mark a step stale")
    inv.add_argument("--step", required=True, choices=STEP_NAMES)

    # status
    sub.add_parser("status", help="Show workflow state")

    # context
    sub.add_parser("context", help="Emit machine-readable context for LLM")

    # run
    run = sub.add_parser("run", help="Run a Makefile target and journal it")
    run.add_argument("--step", default=None, choices=STEP_NAMES)
    run.add_argument("--target", default=None)

    # run-all
    sub.add_parser("run-all", help="Run all stale pipeline steps")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "record":
        cmd_record(args)
    elif args.command == "invalidate":
        cmd_invalidate(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "context":
        cmd_context(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "run-all":
        cmd_run_all(args)


if __name__ == "__main__":
    main()
