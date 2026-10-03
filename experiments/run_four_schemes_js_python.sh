#!/usr/bin/env bash
# Validation-only screening of schemes A-D on JavaScript and Python.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

seed=123456
base_model=${BASE_MODEL:-DeepSoftwareAnalytics/CoCoSoDa}
screen_root=${SCREEN_ROOT:-./runs/four_scheme_screening_20260928}
mkdir -p "${screen_root}"

python -c "import torch, transformers" >/dev/null

source_output() {
  case "$1" in
    javascript)
      echo "./experimental_results/01_initial_cocosoda_ua_hn_bm25_global_seed123456/javascript_seed123456_lr8e-6_tau0.03_globalHN_top500_rank50_20260509210036"
      ;;
    python)
      echo "./experimental_results/01_initial_cocosoda_ua_hn_bm25_global_seed123456/python_seed123456_lr8e-6_tau0.03_globalHN_top500_rank50_20260515033702"
      ;;
    *)
      echo "[ERROR] Unsupported language: $1" >&2
      return 1
      ;;
  esac
}

candidate_file() {
  echo "./runs/table3_controls/candidate_source/candidate_pools/ua_hn/fc/seed_123456/${1}_fc_top32.pkl"
}

metadata_file() {
  if [[ "$1" == "javascript" ]]; then
    echo "./runs/js_method_search/inputs/javascript_fc_boundary_metadata.pkl"
  else
    echo "${screen_root}/inputs/${1}_fc_boundary_metadata.pkl"
  fi
}

ensure_metadata() {
  local lang="$1"
  local source_out
  source_out="$(source_output "${lang}")"
  local checkpoint="${source_out}/checkpoint-best-mrr/model.bin"
  local candidates
  candidates="$(candidate_file "${lang}")"
  local metadata
  metadata="$(metadata_file "${lang}")"

  [[ -f "${checkpoint}" ]] || { echo "[ERROR] Missing ${checkpoint}"; exit 1; }
  [[ -f "${candidates}" ]] || { echo "[ERROR] Missing ${candidates}"; exit 1; }
  if [[ -f "${metadata}" ]]; then
    echo "[skip] Boundary metadata exists: ${metadata}"
    return
  fi

  mkdir -p "$(dirname "${metadata}")"
  echo "[metadata] ${lang}"
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python -m refcode.build_failure_boundary_metadata \
    --train-data-file "dataset/${lang}/train.jsonl" \
    --candidate-file "${candidates}" \
    --output-file "${metadata}" \
    --checkpoint-file "${checkpoint}" \
    --model-name-or-path "${base_model}" \
    --config-name "${base_model}" \
    --tokenizer-name "${base_model}" \
    --seed "${seed}" 2>&1 | tee "${screen_root}/${lang}_metadata.log"
}

run_train_scheme() {
  local lang="$1"
  local scheme="$2"
  local use_distill="$3"
  local source_out
  source_out="$(source_output "${lang}")"
  local checkpoint="${source_out}/checkpoint-best-mrr/model.bin"
  local metadata
  metadata="$(metadata_file "${lang}")"
  local output_dir="${screen_root}/${lang}/${scheme}"

  if [[ -f "${output_dir}/checkpoint-best-mrr/model.bin" && -f "${output_dir}/best_validation.json" ]]; then
    echo "[skip] ${lang} ${scheme} already complete"
    return
  fi
  mkdir -p "${output_dir}"
  echo "[train] ${lang} ${scheme}"
  BASE_MODEL="${base_model}" \
  TOKENIZER_NAME="${base_model}" \
  INITIAL_RETRIEVER_OUT="${source_out}" \
  LOAD_MODEL_FILE="${checkpoint}" \
  SELF_MINED_IDX_FILE="${metadata}" \
  OUTPUT_DIR="${output_dir}" \
  FAILURE_STRATEGY=boundary_preserve \
  USE_RELIABLE_LOCAL_DISTILLATION="${use_distill}" \
  LOCAL_DISTILL_W=${LOCAL_DISTILL_W:-0.05} \
  LOCAL_DISTILL_SAMPLE_SIZE=${LOCAL_DISTILL_SAMPLE_SIZE:-16} \
  LOCAL_DISTILL_EVERY_N_STEPS=${LOCAL_DISTILL_EVERY_N_STEPS:-2} \
  USE_LITE_LATE_INTERACTION_TRAIN=0 \
  LI_W=0 \
  USE_VALID_FUSION_SELECT=0 \
  SELF_MINE_TOPK=8 \
  SELF_MINE_TRAIN_K=1 \
  SKIP_FINAL_RERANK=1 \
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
  bash ./run_refcode.sh --lang "${lang}" --seed "${seed}" \
    2>&1 | tee "${output_dir}/launcher.log"
}

run_structural_schemes() {
  local lang="$1"
  local checkpoint="$2"
  local output_dir="${screen_root}/${lang}/structural"
  local output_file="${output_dir}/validation.json"
  if [[ -f "${output_file}" ]]; then
    echo "[skip] ${lang} C/D structural validation already complete"
    return
  fi
  mkdir -p "${output_dir}"
  echo "[validate] ${lang} C candidate-difference + D compact-multivector"
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python -m refcode.evaluate_structural_variants \
    --loaded-model-filename "${checkpoint}" \
    --eval-data-file "dataset/${lang}/valid.jsonl" \
    --codebase-file "dataset/${lang}/codebase.jsonl" \
    --output-file "${output_file}" \
    --model-name-or-path "${base_model}" \
    --config-name "${base_model}" \
    --tokenizer-name "${base_model}" \
    --top-k 50 \
    --num-vectors 4 \
    --alphas 0.25,0.5,0.75 \
    --fp16 1 2>&1 | tee "${output_dir}/validation.log"
}

link_javascript_a() {
  local lang_root="${screen_root}/javascript"
  local target="${repo_root}/runs/js_method_search/03_boundary_preserve"
  local link="${lang_root}/A_boundary_preserve"
  mkdir -p "${lang_root}"
  if [[ ! -e "${link}" && ! -L "${link}" ]]; then
    ln -s "${target}" "${link}"
  fi
}

echo "[protocol] validation-only, seed=${seed}, no CTRD, source=UA-HN"

# JavaScript: A is already complete, so screen C/D first and then train B.
ensure_metadata javascript
link_javascript_a
javascript_a="${screen_root}/javascript/A_boundary_preserve/checkpoint-best-mrr/model.bin"
run_structural_schemes javascript "${javascript_a}"
run_train_scheme javascript B_local_distillation 1

# Python: train A from the same source, evaluate C/D on A, then train B.
ensure_metadata python
run_train_scheme python A_boundary_preserve 0
python_a="${screen_root}/python/A_boundary_preserve/checkpoint-best-mrr/model.bin"
run_structural_schemes python "${python_a}"
run_train_scheme python B_local_distillation 1

python experiments/summarize_four_schemes.py \
  --root "${screen_root}" \
  --output "${screen_root}/summary.json"

echo "[done] ${screen_root}/summary.json"
