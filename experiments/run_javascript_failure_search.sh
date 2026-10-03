#!/usr/bin/env bash
# Validation-only JavaScript screening for source-boundary-aware FC variants.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

seed=123456
lang=javascript
base_model=${BASE_MODEL:-DeepSoftwareAnalytics/CoCoSoDa}
source_out=./experimental_results/01_initial_cocosoda_ua_hn_bm25_global_seed123456/javascript_seed123456_lr8e-6_tau0.03_globalHN_top500_rank50_20260509210036
source_checkpoint="${source_out}/checkpoint-best-mrr/model.bin"
candidate_file=./runs/table3_controls/candidate_source/candidate_pools/ua_hn/fc/seed_123456/javascript_fc_top32.pkl
search_root=${SEARCH_ROOT:-./runs/js_method_search}
metadata_file="${search_root}/inputs/javascript_fc_boundary_metadata.pkl"

[[ -f "${source_checkpoint}" ]] || { echo "[ERROR] Missing source checkpoint: ${source_checkpoint}"; exit 1; }
[[ -f "${candidate_file}" ]] || { echo "[ERROR] Missing FC candidate pool: ${candidate_file}"; exit 1; }
mkdir -p "${search_root}/inputs"

if [[ ! -f "${metadata_file}" ]]; then
  echo "[metadata] Scoring gold and FC candidates with the frozen source retriever"
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python -m refcode.build_failure_boundary_metadata \
    --train-data-file "dataset/${lang}/train.jsonl" \
    --candidate-file "${candidate_file}" \
    --output-file "${metadata_file}" \
    --checkpoint-file "${source_checkpoint}" \
    --model-name-or-path "${base_model}" \
    --config-name "${base_model}" \
    --tokenizer-name "${base_model}" \
    --seed "${seed}"
else
  echo "[metadata] Reusing ${metadata_file}"
fi

run_variant() {
  local variant="$1"
  local strategy="$2"
  local output_dir="${search_root}/${variant}"

  if [[ -f "${output_dir}/best_validation.json" && -f "${output_dir}/checkpoint-best-mrr/model.bin" ]]; then
    echo "[skip] ${variant} is already complete"
    return
  fi

  mkdir -p "${output_dir}"
  {
    echo "variant=${variant}"
    echo "failure_strategy=${strategy}"
    echo "initial=UA-HN_no_CTRD"
    echo "source_checkpoint=${source_checkpoint}"
    echo "candidate_file=${candidate_file}"
    echo "boundary_metadata=${metadata_file}"
    echo "seed=${seed}"
    echo "selection=fixed validation fusion top50 alpha0.5"
    echo "test_policy=winner_only"
  } > "${output_dir}/provenance.txt"

  echo "[run] ${variant}: FAILURE_STRATEGY=${strategy}"
  BASE_MODEL="${base_model}" \
  TOKENIZER_NAME="${base_model}" \
  INITIAL_RETRIEVER_OUT="${source_out}" \
  LOAD_MODEL_FILE="${source_checkpoint}" \
  SELF_MINED_IDX_FILE="${metadata_file}" \
  OUTPUT_DIR="${output_dir}" \
  FAILURE_STRATEGY="${strategy}" \
  USE_LITE_LATE_INTERACTION_TRAIN=0 \
  LI_W=0 \
  USE_VALID_FUSION_SELECT=1 \
  VALID_FUSION_TOPK=50 \
  VALID_FUSION_ALPHA=0.5 \
  SELF_MINE_TOPK=8 \
  SELF_MINE_TRAIN_K=1 \
  SKIP_FINAL_RERANK=1 \
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
  bash ./run_refcode.sh --lang "${lang}" --seed "${seed}" 2>&1 | tee "${output_dir}/launcher.log"
}

# Each run starts independently from the same source checkpoint.
run_variant 00_baseline baseline
run_variant 01_failure_only failure_only
run_variant 02_boundary boundary
run_variant 03_boundary_preserve boundary_preserve

best_dir="$(python experiments/select_best_validation.py \
  --root "${search_root}" \
  --summary-file "${search_root}/selection_summary.json")"
best_checkpoint="${best_dir}/checkpoint-best-mrr/model.bin"

echo "[select] Validation winner: ${best_dir}"
if [[ ! -f "${best_dir}/result.jsonl" ]]; then
  echo "[test] Evaluating only the validation winner"
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python -m refcode.run_refcode_refinement \
    --run_rerank_only \
    --lang "${lang}" \
    --model_name_or_path "${base_model}" \
    --config_name "${base_model}" \
    --tokenizer_name "${base_model}" \
    --loaded_model_filename "${best_checkpoint}" \
    --eval_data_file "dataset/${lang}/test.jsonl" \
    --codebase_file "dataset/${lang}/codebase.jsonl" \
    --output_dir "${best_dir}" \
    --code_length 256 \
    --nl_length 128 \
    --eval_batch_size 128 \
    --rerank_batch_size 64 \
    --top_k 50 \
    --fusion_alpha 0.5 \
    --rerank_fp16 1 2>&1 | tee "${best_dir}/winner_test.log"
else
  echo "[skip] Winner test result already exists: ${best_dir}/result.jsonl"
fi

echo "[done] Summary: ${search_root}/selection_summary.json"
