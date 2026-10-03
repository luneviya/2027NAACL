#!/usr/bin/env bash
# Rerank one saved ReFCode checkpoint across K and alpha without retraining.
set -euo pipefail

lang=javascript
checkpoint=
output_root=./runs/rerank_sweeps
ks=10,20,50,100
alphas=0,0.25,0.5,0.75,1.0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --lang) lang="$2"; shift 2 ;;
    --checkpoint) checkpoint="$2"; shift 2 ;;
    --output-root) output_root="$2"; shift 2 ;;
    --ks) ks="$2"; shift 2 ;;
    --alphas) alphas="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done
[[ -f "${checkpoint}" ]] || { echo "--checkpoint must point to model.bin"; exit 1; }

IFS=',' read -r -a k_list <<< "${ks}"
IFS=',' read -r -a alpha_list <<< "${alphas}"
for k in "${k_list[@]}"; do
  for alpha in "${alpha_list[@]}"; do
    out="${output_root}/${lang}/top${k}_alpha${alpha}"
    mkdir -p "${out}"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python -m refcode.run_refcode_refinement \
      --run_rerank_only --lang "${lang}" --model_name_or_path "${BASE_MODEL:-DeepSoftwareAnalytics/CoCoSoDa}" \
      --config_name "${BASE_MODEL:-DeepSoftwareAnalytics/CoCoSoDa}" \
      --tokenizer_name "${TOKENIZER_NAME:-${BASE_MODEL:-DeepSoftwareAnalytics/CoCoSoDa}}" \
      --loaded_model_filename "${checkpoint}" --eval_data_file "dataset/${lang}/test.jsonl" \
      --codebase_file "dataset/${lang}/codebase.jsonl" --output_dir "${out}" \
      --code_length 256 --nl_length 128 --eval_batch_size 128 --rerank_batch_size 64 \
      --top_k "${k}" --fusion_alpha "${alpha}" --rerank_fp16 1 | tee "${out}/rerank.log"
  done
done
