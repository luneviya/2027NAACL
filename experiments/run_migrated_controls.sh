#!/usr/bin/env bash
# Run FC/static/random controls from one of the verified migrated Initial runs.
set -euo pipefail

initial=ua_hn
condition=random
seed=123456
langs=ruby,javascript,java,go,php,python
output_root=./runs/migrated_controls

while [[ $# -gt 0 ]]; do
  case "$1" in
    --initial) initial="$2"; shift 2 ;;
    --condition) condition="$2"; shift 2 ;;
    --seed) seed="$2"; shift 2 ;;
    --langs) langs="$2"; shift 2 ;;
    --output-root) output_root="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

case "${initial}" in
  ua_hn)
    initial_root=./experimental_results/01_initial_cocosoda_ua_hn_bm25_global_seed123456
    ;;
  *) echo "--initial must be ua_hn in the NAACL plan"; exit 1 ;;
esac
[[ "${seed}" == "123456" ]] || { echo "Migrated Initial runs only exist for seed 123456."; exit 1; }

IFS=',' read -r -a lang_list <<< "${langs}"
for lang in "${lang_list[@]}"; do
  matches=("${initial_root}/${lang}_seed${seed}_"*)
  [[ ${#matches[@]} -eq 1 ]] || { echo "Expected one Initial directory for ${lang}, found ${#matches[@]}"; exit 1; }
  extra=()
  if [[ "${condition}" == "static" ]]; then
    extra=(--candidate-file "./experimental_inputs/static_hn_top32/${lang}_static_hard_top32.pkl")
  fi
  bash experiments/run_refinement_from_initial.sh --lang "${lang}" --seed "${seed}" \
    --initial-name "${initial}" --initial-out "${matches[0]}" --condition "${condition}" \
    --output-root "${output_root}" "${extra[@]}"
done
