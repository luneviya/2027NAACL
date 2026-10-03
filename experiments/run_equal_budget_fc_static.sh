#!/usr/bin/env bash
# Existing static-HN pools contain one candidate per query. This runner makes
# the FC arm use one candidate too, preventing a candidate-budget confound.
set -euo pipefail

initial=ua_hn
seed=123456
langs=ruby,javascript,java,go,php,python
output_root=./runs/equal_budget_k1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --initial) initial="$2"; shift 2 ;;
    --seed) seed="$2"; shift 2 ;;
    --langs) langs="$2"; shift 2 ;;
    --output-root) output_root="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

# These variables are consumed by run_refcode.sh, invoked by the control runner.
export SELF_MINE_TOPK=1
export SELF_MINE_TRAIN_K=1

bash experiments/run_migrated_controls.sh --initial "${initial}" --condition fc --seed "${seed}" \
  --langs "${langs}" --output-root "${output_root}"
bash experiments/run_migrated_controls.sh --initial "${initial}" --condition static --seed "${seed}" \
  --langs "${langs}" --output-root "${output_root}"
