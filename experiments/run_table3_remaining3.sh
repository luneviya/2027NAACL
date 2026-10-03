#!/usr/bin/env bash
# Run the three unfinished Table 3 configurations sequentially.
# The completed equal-budget FC (K=1, LI off) run is intentionally not repeated.
set -euo pipefail

langs=ruby,javascript,java,go,php,python
seed=123456
output_root=./runs/table3_controls

while [[ $# -gt 0 ]]; do
  case "$1" in
    --langs) langs="$2"; shift 2 ;;
    --seed) seed="$2"; shift 2 ;;
    --output-root) output_root="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

[[ "${seed}" == "123456" ]] || {
  echo "This runner uses the verified migrated UA-HN source for seed 123456."
  exit 1
}

# 1/3: equal-budget Static HN control.
export USE_LITE_LATE_INTERACTION_TRAIN=0
export LI_W=0
export SELF_MINE_TOPK=1
export SELF_MINE_TRAIN_K=1
echo "[1/3] Starting Static HN (K=1, LI off)"
bash experiments/run_migrated_controls.sh --initial ua_hn --condition static \
  --seed "${seed}" --langs "${langs}" \
  --output-root "${output_root}/candidate_source"

# 2/3: equal-budget Random control.
echo "[2/3] Starting Random (K=1, LI off)"
bash experiments/run_migrated_controls.sh --initial ua_hn --condition random \
  --seed "${seed}" --langs "${langs}" \
  --output-root "${output_root}/candidate_source"

# 3/3: normal FC budget without local-interaction training.
export SELF_MINE_TOPK=8
export SELF_MINE_TRAIN_K=1
echo "[3/3] Starting FC no-LI ablation (mining top-k=8)"
bash experiments/run_migrated_controls.sh --initial ua_hn --condition fc \
  --seed "${seed}" --langs "${langs}" \
  --output-root "${output_root}/components/fc_no_li"

echo "All three remaining Table 3 configurations completed."
