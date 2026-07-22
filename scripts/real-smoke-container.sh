#!/usr/bin/env bash
# Run the real-TensorRT smoke corpus inside an official NGC TensorRT
# container, using the INSTALLED trtcheck wheel (never editable mode).
#
# GPU access strategy: prefers the NVIDIA Container Toolkit (--gpus all)
# when Docker advertises the nvidia runtime; otherwise falls back to
# manual passthrough (--device /dev/nvidia* + read-only mounts of the
# driver's user-space libraries), which requires no root and changes
# nothing on the host.
#
# Usage: scripts/real-smoke-container.sh [IMAGE] [WHEEL]
#   IMAGE  default: nvcr.io/nvidia/tensorrt:24.08-py3  (TensorRT 10.3 —
#          matches the repository's 10.3 support target)
#   WHEEL  default: newest dist/trtcheck-*.whl
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="${1:-nvcr.io/nvidia/tensorrt:24.08-py3}"
WHEEL="${2:-$(ls -t "$REPO"/dist/trtcheck-*.whl | head -1)}"
WHEEL="$(readlink -f "$WHEEL")"
OUT="$(mktemp -d "${TMPDIR:-/tmp}/trtcheck-realsmoke.XXXXXX")"
trap 'rm -rf "$OUT/nvlibs"' EXIT
echo "image:  $IMAGE"
echo "wheel:  $WHEEL"
echo "outdir: $OUT"

GPU_ARGS=()
if docker info 2>/dev/null | grep -q "Runtimes:.*nvidia"; then
  GPU_ARGS+=(--gpus all)
else
  echo "nvidia runtime not configured -- using manual GPU passthrough"
  mkdir -p "$OUT/nvlibs"
  # Only the driver-side libraries the CUDA/TensorRT stack dlopens; the
  # container keeps its own CUDA runtime. Read-only.
  ( cd /usr/lib/x86_64-linux-gnu && cp -a \
      libcuda.so* libcudadebugger.so* libnvidia-ml.so* libnvidia-cfg.so* \
      libnvidia-nvvm.so* libnvidia-ptxjitcompiler.so* libnvidia-gpucomp.so* \
      "$OUT/nvlibs/" 2>/dev/null )
  for dev in /dev/nvidia0 /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools; do
    [ -e "$dev" ] && GPU_ARGS+=(--device "$dev")
  done
  GPU_ARGS+=(-v "$OUT/nvlibs:/nvlibs:ro" -e LD_LIBRARY_PATH=/nvlibs)
  # NGC images do not ship nvidia-smi; the toolkit normally injects it.
  [ -x /usr/bin/nvidia-smi ] && GPU_ARGS+=(-v /usr/bin/nvidia-smi:/usr/bin/nvidia-smi:ro)
fi

SQUEEZENET_ARG=""
if [ -f "$REPO/bench/cache/squeezenet1_1.onnx" ]; then
  SQUEEZENET_ARG="--squeezenet /repo/bench/cache/squeezenet1_1.onnx"
fi

docker run --rm "${GPU_ARGS[@]}" \
  -v "$REPO:/repo:ro" \
  -v "$WHEEL:/wheel/$(basename "$WHEEL"):ro" \
  -v "$OUT:/out" \
  -w /out \
  "$IMAGE" \
  bash -lc "
    set -euo pipefail
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
    TRTEXEC=\$(command -v trtexec || ls /usr/src/tensorrt/bin/trtexec /opt/tensorrt/bin/trtexec 2>/dev/null | head -1)
    echo \"trtexec: \$TRTEXEC\"
    \$TRTEXEC --version 2>&1 | tail -1 || true
    pip install --quiet /wheel/*.whl
    trtcheck --version
    python3 /repo/scripts/real_tensorrt_smoke.py \
      --trtexec \"\$TRTEXEC\" --fixtures /repo/tests/fixtures --out /out \
      $SQUEEZENET_ARG
  "
echo
echo "results: $OUT/real_tensorrt_smoke_results.json"
