#!/usr/bin/env bash
# Print PR-changed ONNX files (added or modified) one per line.
#
# Inputs (env):
#   INPUT_PATHS       glob filter, default '**/*.onnx'
#   INPUT_BASE_REF    base ref to diff against; falls back to GITHUB_BASE_SHA
#   GITHUB_BASE_SHA   github.event.pull_request.base.sha
set -euo pipefail

base="${INPUT_BASE_REF:-${GITHUB_BASE_SHA:-}}"
glob="${INPUT_PATHS:-**/*.onnx}"

if [[ -z "$base" ]]; then
    echo "diff_changed_files: no base ref available; emitting nothing" >&2
    exit 0
fi

# Validate the base ref before it ever reaches git. It is concatenated into
# "$base"...HEAD and passed to git fetch/rev-parse, so a value beginning with
# '-' (e.g. --output=...) or containing shell/refspec metacharacters would be
# an option-injection / arbitrary-write primitive. Require a conservative
# git-ref/sha shape that starts with an alphanumeric (no leading dash). Also
# reject any '..' sequence so the value can never read as a path-traversal /
# range-spec trick even if a future use treats it as a path.
if [[ "$base" == *".."* ]] || [[ ! "$base" =~ ^[A-Za-z0-9][A-Za-z0-9._/-]*$ ]]; then
    echo "diff_changed_files: refusing unsafe base ref: $base" >&2
    exit 2
fi

# Make sure we have the base commit locally; in actions/checkout the default
# fetch-depth=1 means earlier commits are missing.
if ! git rev-parse --quiet --verify "$base^{commit}" >/dev/null 2>&1; then
    git fetch --no-tags --depth=50 origin "$base" >/dev/null 2>&1 || true
fi

# Added or modified files since base.
# Validate glob characters before passing to Python. Reject anything that
# could be a shell meta-character escape; we accept standard glob syntax.
if [[ ! "$glob" =~ ^[][A-Za-z0-9._/*?-]+$ ]]; then
    echo "diff_changed_files: refusing unsafe paths glob: $glob" >&2
    exit 2
fi

git diff --name-only --diff-filter=AM "$base"...HEAD \
    | TRTCHECK_GLOB="$glob" python3 -c '
import fnmatch, os, sys
pat = os.environ["TRTCHECK_GLOB"]
for line in sys.stdin:
    name = line.rstrip()
    if name and fnmatch.fnmatch(name, pat):
        print(name)
' 
