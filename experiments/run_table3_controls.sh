#!/usr/bin/env bash
# Controlled experiments for Table 3.  All runs begin from the migrated
# no-CTRD UA-HN source retriever.  No CoCoSoDa or CTRD training is launched.
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

# Panel A: candidate-source comparison.  LI and fusion are not used to define
# the reported metric; all explicit-negative conditions use one candidate.
export USE_LITE_LATE_INTERACTION_TRAIN=0
export LI_W=0
export SELF_MINE_TOPK=1
export SELF_MINE_TRAIN_K=1
bash experiments/run_equal_budget_fc_static.sh --initial ua_hn --seed "${seed}" \
  --langs "${langs}" --output-root "${output_root}/candidate_source"
bash experiments/run_migrated_controls.sh --initial ua_hn --condition random --seed "${seed}" \
  --langs "${langs}" --output-root "${output_root}/candidate_source"

# Panel B: the selected FC pipeline.  Use the normal FC budget, first without
# LI and then with LI.  Global metrics are taken from the training evaluation;
# the final rerank output is used only for the fusion row.
export SELF_MINE_TOPK=8
export SELF_MINE_TRAIN_K=1
export USE_LITE_LATE_INTERACTION_TRAIN=0
export LI_W=0
bash experiments/run_migrated_controls.sh --initial ua_hn --condition fc --seed "${seed}" \
  --langs "${langs}" --output-root "${output_root}/components/fc_no_li"

export USE_LITE_LATE_INTERACTION_TRAIN=1
export LI_W=0.05
bash experiments/run_migrated_controls.sh --initial ua_hn --condition fc --seed "${seed}" \
  --langs "${langs}" --output-root "${output_root}/components/fc_with_li"
