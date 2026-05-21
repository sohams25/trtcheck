# trtcheck validation harness

This directory ships the scaffolding for measuring how accurate trtcheck's
predictions are against real TensorRT conversion outcomes. The scoring
pipeline runs anywhere; the actual `trtexec` invocations need a GPU and a
working TensorRT install, so that step is left to whoever has the
hardware.

## What lives here

| File | Purpose |
|---|---|
| `manifest.yaml` | List of ONNX models with `expected` outcomes (the ground truth). Some entries point at public URLs; some point at the bundled fixtures so the pipeline can be exercised offline. |
| `fetch.py` | Downloads URL-source entries into `bench/cache/` (gitignored). Skips files that already exist. Verifies SHA-256 when listed. |
| `score.py` | Pure scoring function plus a CLI. Takes the manifest and an outcomes JSON, prints a confusion matrix with precision / recall / F1. |
| `cache/` | Gitignored download cache. Created on first fetch. |

The manifest schema is enforced by `tools/validate_bench_manifest.py`,
covered by `tests/test_bench_manifest.py`. The scoring logic is covered by
`tests/test_bench_score.py`.

## End-to-end flow (GPU required)

```bash
# 1. fetch the public models. bundled-fixture rows are skipped automatically.
python bench/fetch.py

# 2. run trtcheck against every model in the manifest and capture predictions.
#    Produce an outcomes file matching this shape:
#
#      {
#        "predictions": {
#          "<manifest entry name>": {
#            "trtcheck": "convert" | "fail",
#            "trtexec":  "convert" | "fail"   # optional, see below
#          },
#          ...
#        }
#      }
#
#    A simple wrapper:

python - <<'PY'
import json, subprocess, yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
with open(ROOT / "bench" / "manifest.yaml") as f:
    entries = yaml.safe_load(f)["models"]

predictions = {}
for e in entries:
    if e["source"].startswith(("http://", "https://")):
        model = ROOT / "bench" / "cache" / f"{e['name']}.onnx"
    else:
        model = ROOT / e["source"]
    out = subprocess.run(
        ["trtcheck", str(model), "--format", "json", "--severity", "critical"],
        capture_output=True, text=True,
    )
    data = json.loads(out.stdout)
    predictions[e["name"]] = {
        "trtcheck": "convert" if data["conversion_likely"] else "fail",
    }
print(json.dumps({"predictions": predictions}, indent=2))
PY
# Pipe the JSON above into bench/outcomes.json.

# 3. run trtexec against the same models on a GPU host. For each, record
#    whether the engine build succeeded (`convert`) or failed (`fail`).
#    Merge those into the outcomes file under the same model keys:
#      predictions.<name>.trtexec = "convert" | "fail"
#
#    The CLI form looks like:
#      trtexec --onnx=<model> --noTF32 ...
#    Failure is any non-zero exit or "Failed to build engine" in stderr.

# 4. score the predictions against the manifest.
python bench/score.py --outcomes bench/outcomes.json
```

## What the score means

`score.py` treats `fail` as the positive class. Each model lands in one of
four cells:

| trtcheck | manifest expected | cell | reading |
|---|---|---|---|
| fail | fail | true positive | trtcheck caught a real failure |
| fail | convert | false positive | trtcheck cried wolf |
| convert | fail | false negative | trtcheck missed a failure |
| convert | convert | true negative | trtcheck correctly stayed quiet |

`precision = TP / (TP + FP)` -- how often "trtcheck says fail" is right.

`recall = TP / (TP + FN)` -- how often a real failure is caught.

If the `trtexec` field disagrees with the manifest's `expected` for an
entry, the entry shows up in the report's `drift` section. That means the
manifest label has gone stale and needs re-checking. Drifted rows don't
affect the confusion matrix; the manifest's `expected` is always treated
as ground truth.

## Where to expand

`manifest.yaml` is intentionally small. Anyone running a corpus expansion
should:

1. Pick models from the ONNX Model Zoo (or anywhere with stable URLs).
2. Run `trtexec` against each on the target TRT version, record the
   outcome, then set the manifest's `expected` to match.
3. Re-run the validator: `python tools/validate_bench_manifest.py`.
4. Open a PR; the manifest grows over time.
