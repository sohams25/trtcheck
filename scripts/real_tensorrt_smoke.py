#!/usr/bin/env python3
"""Real-TensorRT smoke runner: the bounded 7-model corpus, executed against a
genuine trtexec.

Designed to run INSIDE a TensorRT container (or any machine with trtexec)
with the *installed* trtcheck wheel — never editable mode. It runs, per
model: static analysis, ``trtcheck --verify-runtime``, and an independent
direct ``trtexec`` invocation, then records agreement. The dynamic-shape
fixture additionally gets a with-profiles trtexec leg.

Usage (see scripts/real-smoke-container.sh for the container wrapper):

    real_tensorrt_smoke.py --trtexec /path/to/trtexec \
        --fixtures <repo>/tests/fixtures --out /out [--timeout 600]

Writes ``real_tensorrt_smoke_results.json`` into --out. Engines are never
saved (--saveEngine is not passed); all scratch stays under --out.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

_TAIL = 1500

# (name, fixture-relative path, expected direct-trtexec outcome)
CORPUS = [
    ("clean_minimal", "clean_minimal.onnx", "build_success"),
    ("squeezenet1_1_public", "SQUEEZENET", "build_success"),
    ("sequence_empty", "failing/sequence_empty.onnx", "parser_failure"),
    # TensorRT 10.3 trtexec does NOT fail on a dynamic model without shape
    # flags: it warns and auto-overrides every unspecified dynamic dim to 1
    # (observed 2026-07-22; version-specific tool behavior). The explicit
    # with-profile leg below covers the real dynamic path.
    ("fully_dynamic", "failing/fully_dynamic.onnx", "build_success"),
    ("custom_domain", "custom_domain.onnx", "parser_failure"),
    ("uint8_fixed_via_fix", "FIXED", "build_success"),
    ("reshape_int64_shape", "reshape_int64_shape.onnx", "build_success"),
]

# Explicit optimization profile for the fully_dynamic fixture
# (input 'input', rank 4, all dims symbolic).
DYN_PROFILE = [
    "--minShapes=input:1x1x8x8",
    "--optShapes=input:1x3x64x64",
    "--maxShapes=input:2x3x128x128",
]


def _run(cmd: list[str], timeout: int) -> dict:
    start = time.monotonic()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        rc: int | None = proc.returncode
        out, err, note = proc.stdout, proc.stderr, ""
    except subprocess.TimeoutExpired:
        rc, out, err, note = None, "", "", f"timeout after {timeout}s"
    return {
        "command": " ".join(cmd),
        "returncode": rc,
        "elapsed_s": round(time.monotonic() - start, 2),
        # Full stdout is kept for JSON parsing but never written to the
        # results file; records carry only the bounded tails.
        "stdout_full": out,
        "stdout_tail": out[-_TAIL:],
        "stderr_tail": err[-_TAIL:],
        "note": note,
    }


def _classify_trtexec(res: dict) -> str:
    if res["returncode"] is None:
        return "timeout"
    combined = (res["stdout_tail"] + res["stderr_tail"]).lower()
    if res["returncode"] == 0:
        return "build_success"
    parser_markers = (
        "failed to parse onnx",
        "modelimporter",
        "onnx2trt",
        "could not parse",
        "in function importmodel",
        "invalidnode",
        "getplugincreator could not find plugin",
    )
    if any(m in combined for m in parser_markers):
        return "parser_failure"
    return "build_failure"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trtexec", required=True)
    ap.add_argument("--fixtures", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--target-trt", default="10.3")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument(
        "--squeezenet",
        type=Path,
        default=None,
        help="Path to a cached public squeezenet ONNX (skipped if absent).",
    )
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    trtcheck = shutil.which("trtcheck")
    assert trtcheck, "installed trtcheck console script not found on PATH"
    ver = _run([args.trtexec, "--version"], 60)
    # trtexec --version exits non-zero on some builds; the banner still
    # carries "[TensorRT vNNNNN]".
    match = re.search(r"TensorRT v\d+", ver["stdout_full"] + ver["stderr_tail"])
    tensorrt_version = match.group(0) if match else "unknown"

    # Produce the --fix corpus entry with the installed wheel.
    fixed = args.out / "uint8_fixed.onnx"
    fixed.unlink(missing_ok=True)
    fix = _run(
        [
            trtcheck,
            str(args.fixtures / "failing/uint8_input.onnx"),
            "--fix",
            "--output",
            str(fixed),
            "--format",
            "json",
        ],
        args.timeout,
    )
    assert fixed.exists(), f"--fix did not write the fixed model: {fix['stderr_tail']}"

    # Reshape regression: the int64 fixer must refuse.
    refuse = subprocess.run(
        [
            trtcheck,
            str(args.fixtures / "reshape_int64_shape.onnx"),
            "--fix",
            "--dry-run",
            "--format",
            "json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    refuse_fixes = [f["fixer"] for f in json.loads(refuse.stdout)["fixes_applied"]]
    assert "int64_to_int32" not in refuse_fixes, "int64 fixer must refuse Reshape shape input"

    results: dict = {
        "generated_by": "scripts/real_tensorrt_smoke.py",
        "tensorrt_version_line": tensorrt_version,
        "target_trt": args.target_trt,
        "reshape_int64_fixer_refusal": {"fixes_applied": refuse_fixes},
        "models": [],
    }

    for name, rel, expected in CORPUS:
        if rel == "FIXED":
            model = fixed
        elif rel == "SQUEEZENET":
            if not args.squeezenet or not args.squeezenet.exists():
                results["models"].append(
                    {"name": name, "skipped": "no cached public model provided"}
                )
                continue
            model = args.squeezenet
        else:
            model = args.fixtures / rel

        static = _run(
            [trtcheck, str(model), "--target-trt", args.target_trt, "--format", "json"],
            args.timeout,
        )
        static_report = (
            json.loads(static["stdout_full"]) if static["returncode"] is not None else {}
        )

        verify = _run(
            [
                trtcheck,
                str(model),
                "--target-trt",
                args.target_trt,
                "--verify-runtime",
                "--trtexec",
                args.trtexec,
                "--verify-timeout",
                str(args.timeout),
                "--format",
                "json",
            ],
            args.timeout + 60,
        )
        verify_report = (
            json.loads(verify["stdout_full"]) if verify["returncode"] is not None else {}
        )
        rv = verify_report.get("runtime_verification") or {}

        direct = _run([args.trtexec, f"--onnx={model}"], args.timeout)
        direct_outcome = _classify_trtexec(direct)

        wrapper_status = rv.get("status", "missing")
        # Agreement: the wrapper's classification must match the independent run.
        agree = {
            "build_success": wrapper_status == "success",
            "parser_failure": wrapper_status == "parser_failure",
            "build_failure": wrapper_status == "build_failure",
            "timeout": wrapper_status == "timeout",
        }.get(direct_outcome, False)

        entry = {
            "name": name,
            "model": model.name,
            "expected_direct_outcome": expected,
            "static_verdict": static_report.get("verdict"),
            "static_rule_ids": sorted({i["rule_id"] for i in static_report.get("issues", [])}),
            "verify_runtime": {
                "status": wrapper_status,
                "verdict_after": verify_report.get("verdict"),
                "runtime_verified": verify_report.get("runtime_verified"),
                "elapsed_s": verify["elapsed_s"],
            },
            "direct_trtexec": {
                "outcome": direct_outcome,
                "returncode": direct["returncode"],
                "elapsed_s": direct["elapsed_s"],
                "diagnostic": (direct["stderr_tail"] or direct["stdout_tail"])[-400:],
            },
            "wrapper_agrees_with_direct": agree,
            "matched_expectation": direct_outcome == expected,
        }

        if name == "fully_dynamic":
            with_profile = _run([args.trtexec, f"--onnx={model}", *DYN_PROFILE], args.timeout)
            entry["direct_trtexec_with_profile"] = {
                "outcome": _classify_trtexec(with_profile),
                "returncode": with_profile["returncode"],
                "elapsed_s": with_profile["elapsed_s"],
                "profile": " ".join(DYN_PROFILE),
            }

        results["models"].append(entry)
        print(
            f"{name:22s} static={entry['static_verdict']!s:10s} "
            f"wrapper={wrapper_status:15s} direct={direct_outcome:15s} "
            f"agree={agree} expected_ok={entry['matched_expectation']}"
        )

    out_file = args.out / "real_tensorrt_smoke_results.json"
    out_file.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {out_file}")

    ran = [m for m in results["models"] if "skipped" not in m]
    successes = [m for m in ran if m["direct_trtexec"]["outcome"] == "build_success"]
    failures = [
        m for m in ran if m["direct_trtexec"]["outcome"] in ("parser_failure", "build_failure")
    ]
    disagreements = [m["name"] for m in ran if not m["wrapper_agrees_with_direct"]]
    print(
        f"\nsummary: {len(ran)} run, {len(successes)} genuine builds, "
        f"{len(failures)} genuine failures, disagreements: {disagreements or 'none'}"
    )
    return 1 if disagreements or not successes or not failures else 0


if __name__ == "__main__":
    sys.exit(main())
