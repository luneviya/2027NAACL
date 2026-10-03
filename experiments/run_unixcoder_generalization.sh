#!/usr/bin/env bash
# Pure UniXcoder -> ReFCode-G/F on all six CodeSearchNet languages.
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

export BACKBONE_NAME=unixcoder
export SOURCE_NAME=pure_unixcoder
export BACKBONE_MODEL=${UNIXCODER_MODEL:-microsoft/unixcoder-base}
export BACKBONE_GENERALIZATION_ROOT=${UNIXCODER_GENERALIZATION_ROOT:-${repo}/runs/backbone_generalization_no_ua/unixcoder}

exec bash "${repo}/experiments/run_cocosoda_generalization.sh"
