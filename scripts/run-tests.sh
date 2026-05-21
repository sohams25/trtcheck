#!/usr/bin/env bash
# Run the trtcheck test suite in a clean environment.
# ROS' system-wide PYTHONPATH/AMENT_PREFIX_PATH leaks into venv sys.path on
# this dev box, so we strip those before invoking pytest.
set -euo pipefail
cd "$(dirname "$0")/.."
exec env -u PYTHONPATH -u AMENT_PREFIX_PATH .venv/bin/pytest "$@"
