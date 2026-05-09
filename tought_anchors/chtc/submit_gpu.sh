#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$repo_root/chtc/logs"
exec condor_submit "$repo_root/chtc/submit_gpu.sub"
