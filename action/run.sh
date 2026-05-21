#!/usr/bin/env bash
# Entrypoint for the composite action. Discovers .onnx files, runs trtcheck
# on each, aggregates the JSON into one envelope, renders the sticky-comment
# markdown, sets GitHub-Actions outputs, and exits per the fail-on policy.
set -euo pipefail

action_dir="$(cd "$(dirname "$0")" && pwd)"
target_trt="${INPUT_TARGET_TRT:-10.3}"
severity="${INPUT_SEVERITY:-warning}"
fail_on="${INPUT_FAIL_ON:-critical}"
paths_glob="${INPUT_PATHS:-**/*.onnx}"
changed_only="${INPUT_CHANGED_ONLY:-true}"

# Discover candidates
# Validate paths_glob early: only allow safe glob characters. This blocks
# command substitution, IFS abuse, and absolute-path escapes from a caller
# passing a hostile inputs.paths.
if [[ ! "$paths_glob" =~ ^[][A-Za-z0-9._/*?-]+$ ]]; then
    echo "trtcheck action: refusing unsafe paths glob: $paths_glob" >&2
    exit 2
fi

if [[ "$changed_only" == "true" ]]; then
    mapfile -t files < <(bash "${action_dir}/diff_changed_files.sh")
else
    # Discover via python fnmatch -- never expand the user-supplied glob in
    # the shell, since unquoted expansion would be subject to word-splitting
    # and command substitution.
    mapfile -t files < <(python3 - "$paths_glob" <<'PY'
import fnmatch, os, sys
pat = sys.argv[1]
for root, _, names in os.walk(".", followlinks=False):
    for name in names:
        rel = os.path.normpath(os.path.join(root, name))
        if rel.startswith("./"):
            rel = rel[2:]
        if fnmatch.fnmatch(rel, pat):
            print(rel)
PY
)
fi

echo "Analyzing ${#files[@]} file(s):"
for f in "${files[@]}"; do echo "  $f"; done

# Per-file analysis. Skip if file doesn't exist (deleted in PR but still in
# diff list of one diff-filter mode; defense in depth).
results_dir="$(mktemp -d)"
echo '{"files": []}' > "${results_dir}/aggregate.json"

python3 - "${results_dir}/aggregate.json" "$target_trt" "$severity" "${files[@]}" <<'PYEOF'
import json
import shlex
import subprocess
import sys
from pathlib import Path

agg_path = Path(sys.argv[1])
target_trt = sys.argv[2]
severity = sys.argv[3]
files = sys.argv[4:]

aggregate = {"files": []}
for f in files:
    p = Path(f)
    if not p.exists():
        continue
    cmd = [
        "trtcheck", str(p),
        "--target-trt", target_trt,
        "--severity", severity,
        "--format", "json",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    # Even on exit code 1 (conversion will fail) the JSON is on stdout.
    try:
        report = json.loads(proc.stdout)
    except json.JSONDecodeError:
        report = {
            "filename": str(p),
            "issues": [],
            "critical_count": 0,
            "warning_count": 0,
            "info_count": 0,
            "conversion_likely": True,
            "estimated_fix_time": "",
            "_error": "trtcheck did not return valid JSON",
            "_stderr": proc.stderr[:500],
        }
    aggregate["files"].append({"path": str(p), "report": report})

agg_path.write_text(json.dumps(aggregate, indent=2))
print(f"wrote {agg_path}")
PYEOF

# Render the sticky-comment markdown
comment_md="${results_dir}/comment.md"
python3 "${action_dir}/render_comment.py" "${results_dir}/aggregate.json" > "$comment_md"

# Compute totals + status
read crit warn status < <(python3 - "${results_dir}/aggregate.json" "$fail_on" <<'PYEOF'
import json, sys
agg = json.loads(open(sys.argv[1]).read())
policy = sys.argv[2]
crit = sum(f["report"].get("critical_count", 0) for f in agg["files"])
warn = sum(f["report"].get("warning_count", 0) for f in agg["files"])
if policy == "critical":
    status = "fail" if crit > 0 else "pass"
elif policy == "warning":
    status = "fail" if (crit > 0 or warn > 0) else "pass"
else:
    status = "pass"
print(crit, warn, status)
PYEOF
)

# Set outputs
{
    echo "report-json=${results_dir}/aggregate.json"
    echo "comment-md=${comment_md}"
    echo "critical-count=${crit}"
    echo "warning-count=${warn}"
    echo "status=${status}"
} >> "${GITHUB_OUTPUT:-/dev/null}"

# Print summary to logs
echo
echo "trtcheck summary: ${crit} critical, ${warn} warning, status=${status}"
echo "(comment markdown at ${comment_md})"

if [[ "$status" == "fail" ]]; then
    exit 1
fi
