#!/usr/bin/env bash
# Re-export the six verified no-CTRD checkpoints with complete rerank metrics.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

base_model=${BASE_MODEL:-DeepSoftwareAnalytics/CoCoSoDa}
output_root=${OUTPUT_ROOT:-./experimental_results/05_refcode_noctrd_main_seed123456}
legacy_root=${LEGACY_ROOT:-${COCOSODA_REPO:-../CoCoSoDa}/saved_models/aaai_ua_cocosoda_selfmine_late_lite_ablation}
langs=(java javascript ruby python php go)

python experiments/export_noctrd_main_results.py --output-root "${output_root#./}"

for lang in "${langs[@]}"; do
  legacy_run="${legacy_root}/${lang}_woCTRD_K1_W0.45_LI0.05"
  checkpoint="${legacy_run}/checkpoint-best-mrr/model.bin"
  audit_dir="${output_root}/${lang}/audited_top50_alpha0.5"
  result_file="${audit_dir}/result.jsonl"
  [[ -f "${checkpoint}" ]] || { echo "[ERROR] Missing ${checkpoint}"; exit 1; }

  if [[ -f "${result_file}" ]]; then
    echo "[skip] Audited main result exists: ${lang}"
    continue
  fi

  mkdir -p "${audit_dir}"
  {
    echo "language=${lang}"
    echo "seed=123456"
    echo "ctrd=false"
    echo "source=CoCoSoDa+UA-HN_no_CTRD"
    echo "refinement=FC_top8_train1_weight0.45+LI_train_weight0.05"
    echo "checkpoint=${checkpoint}"
    echo "split=test"
    echo "top_k=50"
    echo "fusion_alpha=0.5"
  } > "${audit_dir}/provenance.txt"

  echo "[audit] ${lang}: ${checkpoint}"
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python -m refcode.run_refcode_refinement \
    --run_rerank_only \
    --lang "${lang}" \
    --model_name_or_path "${base_model}" \
    --config_name "${base_model}" \
    --tokenizer_name "${base_model}" \
    --loaded_model_filename "${checkpoint}" \
    --eval_data_file "dataset/${lang}/test.jsonl" \
    --codebase_file "dataset/${lang}/codebase.jsonl" \
    --output_dir "${audit_dir}" \
    --code_length 256 \
    --nl_length 128 \
    --eval_batch_size 128 \
    --rerank_batch_size 64 \
    --top_k 50 \
    --fusion_alpha 0.5 \
    --rerank_fp16 1 2>&1 | tee "${audit_dir}/running.log"
done

python experiments/export_noctrd_main_results.py --output-root "${output_root#./}"
echo "[done] Audited no-CTRD main results: ${output_root}"
