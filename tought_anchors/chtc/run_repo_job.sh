#!/bin/bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <repo-tar.gz> <python-bundle.tar.gz|NONE> <command> [args...]" >&2
  exit 2
fi

repo_archive="$1"
python_bundle="$2"
shift 2

cleanup_scratch=0
if [[ -n "${_CONDOR_SCRATCH_DIR:-}" ]]; then
  scratch_dir="$_CONDOR_SCRATCH_DIR"
else
  fallback_tmp="${THOUGHT_ANCHORS_TMPDIR:-${TMPDIR:-}}"
  if [[ -z "$fallback_tmp" || "$fallback_tmp" == "/tmp" || "$fallback_tmp" == /tmp/* ]]; then
    fallback_tmp="${HOME:-$PWD}/.tmp"
  fi
  mkdir -p "$fallback_tmp"
  scratch_dir="$(mktemp -d "$fallback_tmp/thought-anchors-XXXXXX")"
  cleanup_scratch=1
  trap 'rm -rf "$scratch_dir"' EXIT
fi

repo_dir="$scratch_dir/repo"
python_dir="$scratch_dir/python"

# HTCondor usually transfers input files into the scratch root using their basename.
if [[ ! -f "$repo_archive" ]]; then
  repo_archive="$(basename "$repo_archive")"
fi

if [[ "$python_bundle" != "NONE" && ! -f "$python_bundle" ]]; then
  python_bundle="$(basename "$python_bundle")"
fi

mkdir -p "$repo_dir"

echo "[thought-anchors] scratch: $scratch_dir"
echo "[thought-anchors] extracting repo: $repo_archive"
tar -xzf "$repo_archive" -C "$repo_dir"

if [[ "$python_bundle" != "NONE" ]]; then
  mkdir -p "$python_dir"
  echo "[thought-anchors] extracting Python bundle: $python_bundle"
  tar -xzf "$python_bundle" -C "$python_dir"
  export PYTHONPATH="$python_dir${PYTHONPATH:+:$PYTHONPATH}"

  lib_paths=()
  if [[ -d "$python_dir/torch/lib" ]]; then
    lib_paths+=("$python_dir/torch/lib")
  fi

  if [[ -d "$python_dir/nvidia" ]]; then
    while IFS= read -r lib_dir; do
      lib_paths+=("$lib_dir")
    done < <(find "$python_dir/nvidia" -mindepth 2 -maxdepth 2 -type d -name lib | sort)
  fi

  if [[ ${#lib_paths[@]} -gt 0 ]]; then
    lib_path_prefix="$(IFS=:; echo "${lib_paths[*]}")"
    export LD_LIBRARY_PATH="$lib_path_prefix${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
fi

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export XDG_CACHE_HOME="$scratch_dir/.cache"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export SENTENCE_TRANSFORMERS_HOME="$HF_HOME/sentence_transformers"
export MPLCONFIGDIR="$scratch_dir/.config/matplotlib"

mkdir -p \
  "$XDG_CACHE_HOME" \
  "$HF_HOME" \
  "$HUGGINGFACE_HUB_CACHE" \
  "$TRANSFORMERS_CACHE" \
  "$HF_DATASETS_CACHE" \
  "$SENTENCE_TRANSFORMERS_HOME" \
  "$MPLCONFIGDIR"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && ( "$CUDA_VISIBLE_DEVICES" == GPU-* || "$CUDA_VISIBLE_DEVICES" == MIG-* ) ]]; then
  mapped_index="$(
    nvidia-smi --query-gpu=uuid,index --format=csv,noheader 2>/dev/null \
      | awk -F',' -v target="${CUDA_VISIBLE_DEVICES}" '
          {
            gsub(/^[ \t]+|[ \t]+$/, "", $1);
            gsub(/^[ \t]+|[ \t]+$/, "", $2);
            if ($1 == target || "GPU-" $1 == target || "MIG-" $1 == target) {
              print $2;
              exit;
            }
          }
        '
  )"
  if [[ -n "$mapped_index" ]]; then
    export CUDA_VISIBLE_DEVICES="$mapped_index"
  fi
fi

for env_file in "$scratch_dir/job.env" "$repo_dir/.env"; do
  if [[ -f "$env_file" ]]; then
    echo "[thought-anchors] loading environment from $env_file"
    set -a
    # shellcheck disable=SC1090
    source "$env_file"
    set +a
  fi
done

cd "$repo_dir"

echo "[thought-anchors] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[thought-anchors] nvidia-smi -L"
  nvidia-smi -L || true
fi

if command -v python3 >/dev/null 2>&1; then
  echo "[thought-anchors] torch CUDA probe"
  python3 -c 'import torch; print(f"torch={torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()} count={torch.cuda.device_count()}")' || true
fi

echo "[thought-anchors] running: $*"
exec "$@"
