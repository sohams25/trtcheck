# trtcheck

Static pre-flight checker for ONNX to TensorRT conversion.

`trtcheck` reads an ONNX file, runs five independent checkers, and tells
you in seconds whether the model will convert cleanly to a TensorRT engine.
If it will not, the report explains what to fix. It runs anywhere Python
runs -- no TensorRT, no CUDA driver, no GPU required.

## Why

The PyTorch -> ONNX -> TensorRT pipeline fails most of the time on the
last hop. Errors are cryptic; the iteration loop ("export, wait, read a
C++ traceback, google, try again") burns hours per fix.

`trtcheck` predicts the failure modes locally so you correct them before
invoking `trtexec`.

## Quick links

- [Install](install.md) -- `pip install trtcheck`
- [Usage](usage.md) -- CLI flags, examples, CI integration
- [Fixers](fixers.md) -- what `--fix` rewrites and when it refuses
- [Rule registry](rules.md) -- stable finding ids for CI filtering
- [Design: verdicts & fix safety](design/analysis-verdicts-and-fix-safety.md) -- invariants and trust model
- [Operators](operators/index.md) -- per-operator TensorRT support matrix

## What it checks

| Checker | Catches |
|---|---|
| operator support | Ops missing or partial in the target TRT version |
| precision | UINT8 / INT64 / FLOAT64 / STRING / BF16 inputs, INT64 weights, FLOAT64 from a Cast or Constant |
| dynamic shapes | Multiple symbolic dims on inputs |
| control flow | `Loop` with runtime trip count, nested `Loop`, `If`, `Scan` |
| graph structure | Empty outputs, duplicate node names, oversized constants |
