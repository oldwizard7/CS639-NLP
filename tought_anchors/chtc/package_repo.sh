#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_tarball="${1:-$repo_root/chtc/thought-anchors.tar.gz}"
build_root="${CHTC_BUILD_ROOT:-$repo_root/chtc/build}"
tmp_root="${THOUGHT_ANCHORS_TMPDIR:-$build_root/tmp}"

mkdir -p "$(dirname "$output_tarball")" "$tmp_root"
tmp_tarball="$(mktemp "$tmp_root/thought-anchors-XXXXXX.tar.gz")"
trap 'rm -f "$tmp_tarball"' EXIT

tar \
  --exclude=".git" \
  --exclude=".codex" \
  --exclude=".cache" \
  --exclude=".config" \
  --exclude="__pycache__" \
  --exclude=".pytest_cache" \
  --exclude=".mypy_cache" \
  --exclude=".venv" \
  --exclude=".venv-chtc" \
  --exclude="repo" \
  --exclude="math_rollouts" \
  --exclude="mmlu_rollouts" \
  --exclude="results" \
  --exclude="chtc/build" \
  --exclude="chtc/logs" \
  --exclude="chtc/*.tar.gz" \
  -czf "$tmp_tarball" \
  -C "$repo_root" \
  .

mv "$tmp_tarball" "$output_tarball"
trap - EXIT

echo "Created repo bundle at $output_tarball"
