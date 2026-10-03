#!/usr/bin/env bash
# One resumable queue: audited main-result export, then unfinished Table 3.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

echo "[phase 1/2] Exporting audited six-language no-CTRD main results"
bash experiments/run_noctrd_main_audit.sh

echo "[phase 2/2] Continuing the three unfinished Table 3 configurations"
bash experiments/run_table3_remaining3.sh

echo "[done] Main export and Table 3 queue completed"
