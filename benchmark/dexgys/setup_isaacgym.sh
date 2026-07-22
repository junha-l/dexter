#!/usr/bin/env bash
# Build the Isaac Gym env used by the DexGYS grasp success-rate benchmark (Part B).
#
#   bash benchmark/dexgys/setup_isaacgym.sh
#
# Creates .venv-isaacgym at the repo root. Safe to re-run: the download and the
# extraction are skipped once isaacgym/ is present. The downloaded tarball is
# removed after a successful extraction.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT_DIR="$REPO_ROOT/benchmark/dexgys/isaacgym-env"
VENV="$REPO_ROOT/.venv-isaacgym"
TARBALL="/tmp/isaac-gym-preview-4.tar.gz"
ISAACGYM_DIR="$REPO_ROOT/isaacgym"
URL="https://developer.nvidia.com/isaac-gym-preview-4"

command -v uv >/dev/null || {
    echo "error: uv not found. Install it with:" >&2
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
}

# open3d needs system X11/GL libs, which headless servers usually lack.
missing=()
for lib in libX11.so.6 libGL.so.1 libgomp.so.1; do
    ldconfig -p 2>/dev/null | grep -q "$lib" || missing+=("$lib")
done
if [ ${#missing[@]} -gt 0 ]; then
    pkgs="libx11-6 libgl1 libgomp1"
    if [ "$(id -u)" = 0 ] && command -v apt-get >/dev/null; then
        echo "==> installing system libs for open3d ($pkgs)"
        apt-get update -qq && apt-get install -y -qq $pkgs
    else
        echo "warning: missing system libs (${missing[*]}) needed by open3d." >&2
        echo "         install them with: sudo apt-get install -y $pkgs" >&2
    fi
fi

# Isaac Gym is not on PyPI and requires a manual download step, so vendor it once
# into the repo (gitignored) and let uv install it from that path.
if [ ! -d "$ISAACGYM_DIR/python" ]; then
    if [ ! -s "$TARBALL" ]; then
        echo "==> downloading Isaac Gym Preview 4 (~200 MB)"
        mkdir -p "$(dirname "$TARBALL")"
        curl -fL --progress-bar "$URL" -o "$TARBALL.part"
        mv "$TARBALL.part" "$TARBALL"
    fi
    echo "==> extracting to $ISAACGYM_DIR"
    tar xzf "$TARBALL" -C "$REPO_ROOT"
    # Reclaim the ~200 MB tarball now that it is extracted.
    rm -f "$TARBALL"
fi

echo "==> building $VENV"
UV_PROJECT_ENVIRONMENT="$VENV" uv sync --project "$PROJECT_DIR"

"$VENV/bin/python" -c "import isaacgym, torch; print('isaacgym ok, torch', torch.__version__)"
echo "==> done. Run the benchmark with: $VENV/bin/python benchmark/dexgys/success_rate.py ..."
