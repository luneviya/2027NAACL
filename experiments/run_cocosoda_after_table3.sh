#!/usr/bin/env bash
# Preserve Table 3 as the highest-priority GPU job. Start the six-language
# CoCoSoDa reproduction only after that tmux job exits successfully.

set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
table3_session=refcode_noctrd_table3
table3_log=${TABLE3_LOG:-${repo}/runs/noctrd_export_then_table3.log}
poll_seconds=60

echo "[wait] Waiting for tmux session: $table3_session"
while tmux has-session -t "$table3_session" 2>/dev/null; do
  sleep "$poll_seconds"
done

if ! grep -Fq '[done] Main export and Table 3 queue completed' "$table3_log"; then
  echo "[stop] Table 3 did not record successful completion; CoCoSoDa will not start." >&2
  exit 1
fi

echo "[start] Table 3 completed; starting the CoCoSoDa reproduction"
exec bash "${repo}/experiments/run_reproduce_cocosoda.sh"
