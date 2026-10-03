#!/usr/bin/env bash
# Export paired source/global/fusion ranks and analyze query-level uncertainty.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

gpu=${CUDA_VISIBLE_DEVICES:-0}
python_bin=${PYTHON_BIN:-python}
run_root=${RUN_ROOT:-./runs/query_rank_diagnostics_seed123456}
result_root=${RESULT_ROOT:-./experimental_results/09_query_rank_diagnostics_seed123456}
base_model=${BASE_MODEL:-DeepSoftwareAnalytics/CoCoSoDa}
langs=(java javascript ruby python php go)

resolve_source_checkpoint() {
  local lang=$1
  local source_root=./experimental_results/01_initial_cocosoda_ua_hn_bm25_global_seed123456
  local matches=()
  shopt -s nullglob
  matches=("${source_root}/${lang}_seed123456_"*/checkpoint-best-mrr/model.bin)
  shopt -u nullglob
  if [[ ${#matches[@]} -ne 1 ]]; then
    echo "[error] expected exactly one source checkpoint for ${lang}, found ${#matches[@]}" >&2
    return 1
  fi
  printf '%s\n' "${matches[0]}"
}

run_eval() {
  local role=$1
  local lang=$2
  local checkpoint=$3
  local output_dir="${run_root}/${role}/${lang}"
  local extra_args=()
  if [[ "${role}" == source ]]; then
    extra_args+=(--global-only)
  fi
  if [[ -f "${output_dir}/grid_results.json" && -f "${output_dir}/query_ranks.jsonl" ]]; then
    echo "[skip] ${role}/${lang}"
    return
  fi
  [[ -f "${checkpoint}" ]] || { echo "[error] missing ${checkpoint}"; exit 1; }
  mkdir -p "${output_dir}"
  {
    echo "role=${role}"
    echo "language=${lang}"
    echo "split=test"
    echo "seed=123456"
    echo "checkpoint=${checkpoint}"
    echo "fusion_k=20"
    echo "fusion_alpha=0.5"
    echo "selection=validation-frozen"
    echo "candidate_corpus=dataset/${lang}/codebase.jsonl"
  } > "${output_dir}/provenance.txt"
  echo "[run] ${role}/${lang}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u experiments/run_rerank_grid.py \
    --lang "${lang}" \
    --checkpoint "${checkpoint}" \
    --split test \
    --output-dir "${output_dir}" \
    --ks 20 \
    --alphas 0.5 \
    --base-model "${base_model}" \
    --eval-batch-size 128 \
    --rerank-batch-size 64 \
    --code-length 256 \
    --nl-length 128 \
    --fp16 1 \
    --save-per-query \
    "${extra_args[@]}" 2>&1 | tee "${output_dir}/running.log"
}

mkdir -p "${run_root}" "${result_root}"

echo "[phase 1/3] Source global ranks"
for lang in "${langs[@]}"; do
  source_checkpoint=$(resolve_source_checkpoint "${lang}")
  run_eval source "${lang}" "${source_checkpoint}"
done

echo "[phase 2/3] Final FC-global and frozen-fusion ranks"
for lang in "${langs[@]}"; do
  final_checkpoint="./runs/table3_controls/components/fc_no_li/ua_hn/fc/seed_123456/${lang}/checkpoint-best-mrr/model.bin"
  run_eval final "${lang}" "${final_checkpoint}"
done

echo "[phase 3/3] Paired bootstrap and rank-transition analysis"
"${python_bin}" experiments/analyze_query_rank_diagnostics.py \
  --run-root "${run_root#./}" \
  --output-root "${result_root#./}" \
  --bootstrap-repetitions 10000

echo "[done] Query-rank diagnostics completed"
