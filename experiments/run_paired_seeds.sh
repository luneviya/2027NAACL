#!/usr/bin/env bash
# Run matched no-CTRD UA-HN Initial and ReFCode FC+LI pipelines for new seeds.
set -euo pipefail

seeds=2027,2028
langs=ruby,javascript,java,go,php,python
output_root=./runs/paired_seed

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seeds) seeds="$2"; shift 2 ;;
    --langs) langs="$2"; shift 2 ;;
    --output-root) output_root="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

IFS=',' read -r -a seed_list <<< "${seeds}"
IFS=',' read -r -a lang_list <<< "${langs}"
for seed in "${seed_list[@]}"; do
  for lang in "${lang_list[@]}"; do
    initial_root="${output_root}/initial_variants/ua_hn/seed_${seed}/${lang}"
    bash experiments/run_initial_variant.sh --lang "${lang}" --seed "${seed}" \
      --variant ua_hn --output-root "${output_root}/initial_variants"
    bash experiments/run_refinement_from_initial.sh --lang "${lang}" --seed "${seed}" \
      --initial-name ua_hn --initial-out "${initial_root}" --condition fc \
      --output-root "${output_root}/refinement"
  done
done
