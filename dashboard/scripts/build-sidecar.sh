#!/usr/bin/env bash
# Build the dashboard-api PyInstaller binary used by the Tauri
# shell as its `externalBin` sidecar.
#
# Tauri's externalBin convention requires a target-triple
# suffix on the binary name, e.g. dashboard-api-aarch64-apple-darwin
# for Apple Silicon Macs. We auto-detect the triple here.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DASHBOARD_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="$(cd "${DASHBOARD_DIR}/.." && pwd)"
BIN_DIR="${DASHBOARD_DIR}/src-tauri/binaries"

# Resolve target triple. The Rust convention on macOS is
# aarch64-apple-darwin (Apple Silicon) or x86_64-apple-darwin.
ARCH="$(uname -m)"
case "${ARCH}" in
    arm64)  TRIPLE="aarch64-apple-darwin" ;;
    x86_64) TRIPLE="x86_64-apple-darwin"  ;;
    *)      echo "Unsupported arch ${ARCH}"; exit 1 ;;
esac

mkdir -p "${BIN_DIR}"
cd "${REPO_DIR}"

if ! command -v pyinstaller >/dev/null 2>&1; then
    echo "pyinstaller not found. Install with: pip install pyinstaller"
    exit 1
fi

# --paths puts the repo root on sys.path so the dashboard_api
# module can import src.* modules during the freeze.
pyinstaller \
    --onefile \
    --paths "${REPO_DIR}" \
    --name "dashboard-api-${TRIPLE}" \
    --distpath "${BIN_DIR}" \
    --workpath "${BIN_DIR}/.build" \
    --specpath "${BIN_DIR}/.spec" \
    --hidden-import uvicorn.logging \
    --hidden-import uvicorn.loops \
    --hidden-import uvicorn.loops.auto \
    --hidden-import uvicorn.protocols \
    --hidden-import uvicorn.protocols.http \
    --hidden-import uvicorn.protocols.http.auto \
    --hidden-import uvicorn.lifespan \
    --hidden-import uvicorn.lifespan.on \
    src/dashboard_api/server.py

# Also drop a non-triple symlink so a plain `dashboard-api`
# invocation works during dev (Tauri only needs the triple
# version for the bundle).
ln -sf "dashboard-api-${TRIPLE}" "${BIN_DIR}/dashboard-api"

echo
echo "Built: ${BIN_DIR}/dashboard-api-${TRIPLE}"
echo "Symlink: ${BIN_DIR}/dashboard-api -> dashboard-api-${TRIPLE}"
