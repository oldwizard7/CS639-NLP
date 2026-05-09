#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bundle_dir="${1:-$repo_root/chtc/build/python}"
output_tarball="${2:-$repo_root/chtc/python-bundle.tar.gz}"
python_bin="${PYTHON_BIN:-python3}"
build_root="${CHTC_BUILD_ROOT:-$repo_root/chtc/build}"
cache_root="${THOUGHT_ANCHORS_CACHE_ROOT:-$build_root/cache}"

# Keep install temp files and caches under the repo/home directory instead of
# inheriting a submit-node TMPDIR that may point at /tmp.
export TMPDIR="${THOUGHT_ANCHORS_TMPDIR:-$build_root/tmp}"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export PIP_CACHE_DIR="${THOUGHT_ANCHORS_PIP_CACHE_DIR:-$cache_root/pip}"
export XDG_CACHE_HOME="${THOUGHT_ANCHORS_XDG_CACHE_HOME:-$cache_root/xdg}"
export UV_CACHE_DIR="${THOUGHT_ANCHORS_UV_CACHE_DIR:-$cache_root/uv}"

rm -rf "$bundle_dir"
mkdir -p \
  "$bundle_dir" \
  "$(dirname "$output_tarball")" \
  "$TMPDIR" \
  "$PIP_CACHE_DIR" \
  "$XDG_CACHE_HOME" \
  "$UV_CACHE_DIR"

echo "[thought-anchors] TMPDIR=$TMPDIR"
echo "[thought-anchors] PIP_CACHE_DIR=$PIP_CACHE_DIR"
echo "[thought-anchors] XDG_CACHE_HOME=$XDG_CACHE_HOME"

"$python_bin" -m pip install --cache-dir "$PIP_CACHE_DIR" --upgrade pip
"$python_bin" -m pip install --cache-dir "$PIP_CACHE_DIR" --target "$bundle_dir" -r "$repo_root/requirements-chtc.txt"

tar -czf "$output_tarball" -C "$bundle_dir" .

echo "Created Python bundle at $output_tarball"
