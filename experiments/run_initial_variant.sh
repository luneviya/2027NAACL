#!/usr/bin/env bash
# Train the sole primary source retriever: CoCoSoDa + UA-HN.
# CTRD and pure-CoCoSoDa training are intentionally excluded from this NAACL plan.
set -euo pipefail

lang=javascript
seed=123456
variant=ua_hn
output_root=./runs/initial_variants
python_bin=${PYTHON_BIN:-python}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --lang) lang="$2"; shift 2 ;;
    --seed) seed="$2"; shift 2 ;;
    --variant) variant="$2"; shift 2 ;;
    --output-root) output_root="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

case "${variant}" in
  ua_hn) use_ua=1 ;;
  *) echo "--variant must be ua_hn in the NAACL plan"; exit 1 ;;
esac

cuda=${CUDA_VISIBLE_DEVICES:-0}
base_model=${BASE_MODEL:-DeepSoftwareAnalytics/CoCoSoDa}
tokenizer=${TOKENIZER_NAME:-${base_model}}
out="${output_root}/${variant}/seed_${seed}/${lang}"
hard_idx="${output_root}/candidate_pools/${variant}/seed_${seed}/${lang}_bm25_top500_rank50.pkl"
cache="${output_root}/embedding_cache/${lang}.pt"
mkdir -p "${out}" "$(dirname "${hard_idx}")" "$(dirname "${cache}")"

if [[ ! -f "${hard_idx}" ]]; then
  CUDA_VISIBLE_DEVICES="${cuda}" "${python_bin}" -m refcode.run_initial_retriever \
    --train_data_file "dataset/${lang}/train.jsonl" \
    --output_file "${hard_idx}" \
    --model_name_or_path "${base_model}" \
    --config_name "${base_model}" \
    --tokenizer_name "${tokenizer}" \
    --nl_length 128 --encode_batch_size 128 --topk 500 --bm25_rank 50 \
    --chunk_size 512 --seed "${seed}" --device cuda --embeddings_cache "${cache}" \
    2>&1 | tee "${out}/hard_negative_mining.log"
else
  echo "[skip] Existing source hard-negative cache: ${hard_idx}"
fi

args=(
  --eval_frequency 100 --moco_m 0.999 --moco_t 0.07 --model_type base
  --output_dir "${out}" --data_aug_type random_mask --moco_k 1024
  --config_name="${base_model}" --model_name_or_path="${base_model}" --tokenizer_name="${tokenizer}"
  --lang="${lang}" --do_train --do_test
  --train_data_file="dataset/${lang}/train.jsonl" --eval_data_file="dataset/${lang}/valid.jsonl"
  --test_data_file="dataset/${lang}/test.jsonl" --codebase_file="dataset/${lang}/codebase.jsonl"
  --num_train_epochs 10 --code_length 256 --nl_length 128 --train_batch_size 128 --eval_batch_size 64
  --learning_rate 8e-6 --seed "${seed}"
)
if [[ "${use_ua}" -eq 1 ]]; then
  args+=(--use_refcode_uncertainty --refcode_temperature 0.03 --refcode_uncertainty_samples 4
    --refcode_intra_weight 1.0 --refcode_kl_weight 1e-5
    --use_refcode_global_hard_negative --refcode_hard_idx_file "${hard_idx}"
    --refcode_hard_mode batch_all --refcode_hard_weight 1.0)
fi
printf '[Initial variant] variant=%s lang=%s seed=%s\n' "${variant}" "${lang}" "${seed}" | tee "${out}/provenance.txt"
{
  printf 'source_retriever=%s+UA-HN\n' "${base_model}"
  printf 'hard_idx=%s\n' "${hard_idx}"
  printf 'use_ctrd=0\n'
  printf 'use_lite_late_interaction_train=0\n'
} | tee -a "${out}/provenance.txt"

checkpoint_file="${out}/checkpoint-best-mrr/model.bin"
if [[ -f "${checkpoint_file}" && -f "${out}/result.jsonl" ]]; then
  echo "[skip] Completed source run: ${out}"
  exit 0
fi

if [[ -f "${checkpoint_file}" ]]; then
  echo "[resume] Source checkpoint exists; running final global test evaluation only."
  CUDA_VISIBLE_DEVICES="${cuda}" "${python_bin}" -m refcode.run_refcode_refinement \
    --output_dir "${out}" \
    --config_name "${base_model}" \
    --model_name_or_path "${base_model}" \
    --tokenizer_name "${tokenizer}" \
    --loaded_model_filename "${checkpoint_file}" \
    --lang "${lang}" --do_test \
    --test_data_file "dataset/${lang}/test.jsonl" \
    --codebase_file "dataset/${lang}/codebase.jsonl" \
    --code_length 256 --nl_length 128 --eval_batch_size 64 --seed "${seed}" \
    2>&1 | tee "${out}/resume_evaluation.log"
  exit 0
fi

CUDA_VISIBLE_DEVICES="${cuda}" "${python_bin}" -m refcode.run_refcode_refinement \
  "${args[@]}" 2>&1 | tee "${out}/running.log"
