#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

uv run python build_decomp_dataset.py \
  --datasets aime25 \
  --max-depth "${MAX_DEPTH:-1}" \
  --max-workers "${MAX_WORKERS:-4}" \
  "$@"
