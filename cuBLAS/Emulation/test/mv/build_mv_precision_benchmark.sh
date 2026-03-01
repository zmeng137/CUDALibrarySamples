#!/usr/bin/env bash
# Build script for mv_precision_benchmark.cu
#
# Adjust -arch to match your GPU:
#   sm_80  → A100
#   sm_86  → RTX 3090 / A5000 / A6000
#   sm_89  → RTX 4090 / L40
#   sm_90  → H100 / GH200
#   sm_100 → Blackwell (B100/B200)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${SCRIPT_DIR}/mv_precision_benchmark.cu"
OUT="${SCRIPT_DIR}/mv_precision_benchmark"
UTILS_INC="${SCRIPT_DIR}/../../../utils"

nvcc -O2 -std=c++11 \
     -arch=sm_86 \
     -I"${UTILS_INC}" \
     "${SRC}" \
     -lcublas \
     -o "${OUT}"

echo "Build successful: ${OUT}"
echo ""
echo "Run:  ${OUT}"
echo "Plot: python3 ${SCRIPT_DIR}/plot_mv_precision_heatmap.py"
