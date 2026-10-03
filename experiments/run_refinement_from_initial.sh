#!/usr/bin/env bash
# Train ReFCode from a supplied initial checkpoint. Candidate source is explicit
# so FC, static-HN, and random-negative runs cannot overwrite one another.
set -euo pipefail

lang=javascript
seed=123456
initial_name=unspecified_initial
initial_out=
condition=fc
candidate_file=
base_model=${BASE_MODEL:-DeepSoftwareAnalytics/CoCoSoDa}
tokenizer_name=${TOKENIZER_NAME:-${base_model}}
output_root=./runs/refinement_controls
python_bin=${PYTHON_BIN:-python}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --lang) lang="$2"; shift 2 ;;
    --seed) seed="$2"; shift 2 ;;
    --initial-name) initial_name="$2"; shift 2 ;;
    --initial-out) initial_out="$2"; shift 2 ;;
    --condition) condition="$2"; shift 2 ;;
    --candidate-file) candidate_file="$2"; shift 2 ;;
    --output-root) output_root="$2"; shift 2 ;;
    --base-model) base_model="$2"; shift 2 ;;
    --tokenizer-name) tokenizer_name="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

[[ -n "${initial_out}" ]] || { echo "--initial-out is required"; exit 1; }
[[ -f "${initial_out}/checkpoint-best-mrr/model.bin" ]] || { echo "Initial checkpoint missing"; exit 1; }
case "${condition}" in fc|static|random) ;; *) echo "--condition must be fc, static, or random"; exit 1;; esac
run_root="${output_root}/${initial_name}/${condition}/seed_${seed}/${lang}"
candidate_dir="${output_root}/candidate_pools/${initial_name}/${condition}/seed_${seed}"
mkdir -p "${run_root}" "${candidate_dir}"

checkpoint_file="${run_root}/checkpoint-best-mrr/model.bin"
if [[ -f "${checkpoint_file}" && -f "${run_root}/result.jsonl" ]]; then
  echo "[skip] Completed ${condition} run: ${run_root}"
  exit 0
fi

if [[ -f "${checkpoint_file}" ]]; then
  if [[ "${SKIP_FINAL_RERANK:-0}" == "1" ]]; then
    echo "[resume] Checkpoint exists and SKIP_FINAL_RERANK=1; leaving frozen evaluation to the caller: ${run_root}"
    exit 0
  fi
  echo "[resume] Checkpoint exists; running final evaluation only: ${run_root}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "${python_bin}" -m refcode.run_refcode_refinement \
    --run_rerank_only \
    --lang "${lang}" \
    --model_name_or_path "${base_model}" \
    --config_name "${base_model}" \
    --tokenizer_name "${tokenizer_name}" \
    --loaded_model_filename "${checkpoint_file}" \
    --eval_data_file "dataset/${lang}/test.jsonl" \
    --codebase_file "dataset/${lang}/codebase.jsonl" \
    --output_dir "${run_root}" \
    --code_length 256 \
    --nl_length 128 \
    --eval_batch_size 128 \
    --rerank_batch_size 64 \
    --top_k 50 \
    --fusion_alpha 0.5 \
    --rerank_fp16 1 2>&1 | tee "${run_root}/resume_evaluation.log"
  exit 0
fi


if [[ -z "${candidate_file}" ]]; then
  case "${condition}" in
    fc)
      candidate_file="${candidate_dir}/${lang}_fc_top32.pkl"
      if [[ ! -f "${candidate_file}" ]]; then
        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "${python_bin}" -m refcode.run_failure_harvesting \
          --train_data_file "dataset/${lang}/train.jsonl" --output_file "${candidate_file}" \
          --model_name_or_path "${base_model}" \
          --config_name "${base_model}" \
          --tokenizer_name "${tokenizer_name}" \
          --checkpoint_file "${initial_out}/checkpoint-best-mrr/model.bin" --seed "${seed}"
      fi
      ;;
    random)
      candidate_file="${candidate_dir}/${lang}_random_top32.pkl"
      if [[ ! -f "${candidate_file}" ]]; then
        "${python_bin}" -m refcode.build_random_negatives --train-data-file "dataset/${lang}/train.jsonl" \
          --output-file "${candidate_file}" --topk 32 --seed "${seed}"
      fi
      ;;
    static)
      echo "Static HN requires --candidate-file pointing to a row-aligned static candidate pickle."
      exit 1
      ;;
  esac
fi
[[ -f "${candidate_file}" ]] || { echo "Candidate file missing: ${candidate_file}"; exit 1; }

{
  echo "initial_name=${initial_name}"
  echo "initial_out=${initial_out}"
  echo "condition=${condition}"
  echo "candidate_file=${candidate_file}"
  echo "seed=${seed}"
  echo "base_model=${base_model}"
  echo "tokenizer_name=${tokenizer_name}"
  echo "use_refcode_uncertainty=${USE_REFCODE_UNCERTAINTY:-1}"
  echo "use_lite_late_interaction_train=${USE_LITE_LATE_INTERACTION_TRAIN:-1}"
  echo "li_weight=${LI_W:-0.05}"
  echo "rerank=top50_alpha0.5; select alpha on validation before final test reporting"
} > "${run_root}/provenance.txt"

BASE_MODEL="${base_model}" \
TOKENIZER_NAME="${tokenizer_name}" \
INITIAL_RETRIEVER_OUT="${initial_out}" \
LOAD_MODEL_FILE="${initial_out}/checkpoint-best-mrr/model.bin" \
SELF_MINED_IDX_FILE="${candidate_file}" \
OUTPUT_DIR="${run_root}" \
SUPERVISION="${condition}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
bash ./run_refcode.sh --lang "${lang}" --seed "${seed}" 2>&1 | tee "${run_root}/launcher.log"
