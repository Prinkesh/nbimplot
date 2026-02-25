#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WASM_DIR="${ROOT_DIR}/wasm"
BUILD_DIR="${WASM_DIR}/build"

NBIMPLOT_WITH_IMPLOT="${NBIMPLOT_WITH_IMPLOT:-OFF}"
NBIMPLOT_IMGUI_DIR="${NBIMPLOT_IMGUI_DIR:-}"
NBIMPLOT_IMPLOT_DIR="${NBIMPLOT_IMPLOT_DIR:-}"

if [[ "${NBIMPLOT_WITH_IMPLOT}" == "ON" ]]; then
  if [[ -z "${NBIMPLOT_IMGUI_DIR}" && -d "${ROOT_DIR}/third_party/imgui" ]]; then
    NBIMPLOT_IMGUI_DIR="${ROOT_DIR}/third_party/imgui"
  fi
  if [[ -z "${NBIMPLOT_IMPLOT_DIR}" && -d "${ROOT_DIR}/third_party/implot" ]]; then
    NBIMPLOT_IMPLOT_DIR="${ROOT_DIR}/third_party/implot"
  fi
fi

if ! command -v emcmake >/dev/null 2>&1; then
  echo "error: emcmake is required (install Emscripten SDK)." >&2
  exit 1
fi

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

emcmake cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DNBIMPLOT_WITH_IMPLOT="${NBIMPLOT_WITH_IMPLOT}" \
  -DNBIMPLOT_IMGUI_DIR="${NBIMPLOT_IMGUI_DIR}" \
  -DNBIMPLOT_IMPLOT_DIR="${NBIMPLOT_IMPLOT_DIR}"

cmake --build . -j

echo "Built: ${ROOT_DIR}/nbimplot/wasm/nbimplot_wasm.js"
