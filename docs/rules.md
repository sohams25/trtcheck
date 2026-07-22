# Rule registry

Every finding trtcheck emits carries a stable, machine-readable `rule_id`.
Renaming or removing an id is a breaking change guarded by
`tests/test_verdicts.py::test_rule_id_registry_is_stable`; additions are
backward-compatible.

Per-issue metadata (JSON schema 2.0): `rule_id`, `severity`, `category`,
`node_name`, `operator`, `graph_scope` (owning graph, where known),
`target_trt`, `confidence` (`high` / `medium` / `low`), `verify_required`
(true when the finding needs runtime or manual verification), `remediation`,
`docs_link`.

## Operator support

| Rule | Severity | Meaning |
|---|---|---|
| `TRT-OP-UNSUPPORTED` | critical | Operator documented as not supported for the target TRT version. |
| `TRT-OP-PARTIAL` | warning | Partial support with documented limitations; verify against your export. |
| `TRT-OP-UNCLASSIFIED` | info (verify) | Operator not classified in trtcheck's matrix — no evidence either way. One finding per op type. |
| `TRT-OP-CUSTOM-DOMAIN` | info (verify) | Custom-domain op; needs a TensorRT plugin. Suppress with `--plugin-domain`. |
| `TRT-OP-CONDITION` | per-condition | A documented conditional-support rule is violated (e.g. TopK `sorted=0`, Resize `mode=cubic`). |
| `TRT-OP-CONDITION-UNRESOLVED` | info (verify) | A conditional-support rule cannot be settled statically (e.g. runtime-dynamic TopK `K`). |

## Precision / dtype

| Rule | Severity |
|---|---|
| `TRT-DTYPE-UINT8-INPUT` | critical |
| `TRT-DTYPE-STRING` | critical |
| `TRT-DTYPE-FP64` | warning |
| `TRT-DTYPE-INT64-INPUT` | warning |
| `TRT-DTYPE-INT64-WEIGHTS` | warning |
| `TRT-DTYPE-BF16` | warning (verify) |

## Shapes and control flow

| Rule | Severity |
|---|---|
| `TRT-SHAPE-PROFILE-MISSING` | warning (verify) |
| `TRT-CONTROL-LOOP-RUNTIME-TRIP` | critical |
| `TRT-CONTROL-LOOP-DYNAMIC-TRIP` | warning (verify) |
| `TRT-CONTROL-LOOP-NESTED` | critical |
| `TRT-CONTROL-IF-SHAPE-MISMATCH` | critical |
| `TRT-CONTROL-IF-UNVERIFIED` | warning (verify) |
| `TRT-CONTROL-SCAN` | warning (verify) |

## Graph structure

| Rule | Severity |
|---|---|
| `TRT-GRAPH-NO-OUTPUT` | critical |
| `TRT-GRAPH-INPUT-UNTYPED` | critical |
| `TRT-GRAPH-DUP-NODE-NAME` | warning |
| `TRT-GRAPH-ISOLATED-NODE` | warning |
| `TRT-GRAPH-LARGE-CONSTANT` | info |
| `TRT-GRAPH-EXTERNAL-DATA` | critical |
| `TRT-GRAPH-ALIASING` | warning |
| `TRT-OPSET-OLD` | info |

## Plugins

| Rule | Severity |
|---|---|
| `TRT-PLUGIN-CHECKER-ERROR` | warning (verify) — a third-party checker crashed; its coverage is missing from this report. |

Severities for remediation-DB rules come from
`trtcheck/data/remediation_db.json`; the table above summarizes, the JSON is
authoritative.
