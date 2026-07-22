#!/usr/bin/env bash
# Package smoke test: install the built wheel into a FRESH venv outside the
# repository and exercise the public surface from there. Catches packaging
# bugs editable installs hide (missing data files, broken entry points).
#
# Usage: scripts/package-smoke.sh [path-to-wheel]
# Default wheel: the newest dist/trtcheck-*.whl in the repo.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
WHEEL="${1:-$(ls -t "$REPO"/dist/trtcheck-*.whl | head -1)}"
[ -f "$WHEEL" ] && WHEEL="$(readlink -f "$WHEEL")"

WORK="$(mktemp -d "${TMPDIR:-/tmp}/trtcheck-smoke.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
cd "$WORK"   # everything below runs OUTSIDE the repository

python3 -m venv venv
# Keep the smoke honest: no repo paths, no dev env leakage.
env -u PYTHONPATH -u AMENT_PREFIX_PATH ./venv/bin/pip -q install "$WHEEL"

run() { env -u PYTHONPATH -u AMENT_PREFIX_PATH "$@"; }

echo "== import + version"
run ./venv/bin/python -c "import trtcheck; print('trtcheck', trtcheck.__version__)"

echo "== packaged data files"
run ./venv/bin/python - <<'PY'
from importlib import resources
import json
for name in ("operator_matrix.json", "remediation_db.json"):
    data = json.loads(resources.files("trtcheck.data").joinpath(name).read_text())
    assert data.get("schema_version"), name
print("data files load: ok")
PY

echo "== console entry point --help"
run ./venv/bin/trtcheck --help > /dev/null
run ./venv/bin/python -m trtcheck --version

echo "== generate a model and analyze (console + json)"
run ./venv/bin/python - <<'PY'
import numpy as np, onnx
from onnx import TensorProto, helper, numpy_helper
inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [10, 4])
out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [3, 4])
idx = numpy_helper.from_array(np.array([0, 1, 2], dtype=np.int64), name="indices")
gather = helper.make_node("Gather", ["input", "indices"], ["g"], name="g0", axis=0)
drop = helper.make_node("Dropout", ["g"], ["d"], name="drop")
ident = helper.make_node("Identity", ["d"], ["output"], name="ident")
graph = helper.make_graph([gather, drop, ident], "m", [inp], [out], initializer=[idx])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
model.ir_version = 8
onnx.save(model, "smoke.onnx")
print("wrote smoke.onnx")
PY
run ./venv/bin/trtcheck smoke.onnx | head -5
run ./venv/bin/trtcheck smoke.onnx --format json --output report.json
run ./venv/bin/python - <<'PY'
import json
report = json.load(open("report.json"))
assert report["schema_version"] == "2.0"
assert report["verdict"] in ("blocked", "unverified", "likely", "verified")
assert all(i["rule_id"] for i in report["issues"])
print("json report: ok, verdict =", report["verdict"])
PY

echo "== safe fix mode"
run ./venv/bin/trtcheck smoke.onnx --fix --output fixed.onnx | tail -3
run ./venv/bin/python -c "import onnx; onnx.checker.check_model(onnx.load('fixed.onnx'), full_check=True); print('fixed model fully valid')"

echo "== missing-verifier behavior (empty PATH)"
run env PATH="" ./venv/bin/trtcheck smoke.onnx --verify-runtime --format json --output verify.json 2> verify.err || true
grep -q '"status": "missing_trtexec"' verify.json
echo "missing trtexec handled: ok"

echo
echo "PACKAGE SMOKE: PASS ($WHEEL)"
