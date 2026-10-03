#!/usr/bin/env bash
# Clean transfer run: UniXcoder UA-HN -> FC-no-LI -> frozen K20/alpha0.5.
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${repo}"

seed=123456
gpu=${CUDA_VISIBLE_DEVICES:-0}
model=${UNIXCODER_MODEL:-microsoft/unixcoder-base}
python_bin=${PYTHON_BIN:-python}
run_root=${TRANSFER_RUN_ROOT:-./runs/backbone_transfer_no_li/unixcoder}
source_root=${run_root}/source
refinement_root=${run_root}/refinement
evaluation_root=${run_root}/evaluation
status_file=${run_root}/STATUS.txt
langs=(javascript java)

if [[ -d "${model}" && ( ! -f "${model}/config.json" || ! -f "${model}/pytorch_model.bin" ) ]]; then
  echo "Local UniXcoder model is incomplete: ${model}" >&2
  exit 1
fi
mkdir -p "${run_root}" "${evaluation_root}"
printf 'state=running\nseed=%s\nbackbone=%s\n' "${seed}" "${model}" > "${status_file}"

on_exit() {
  code=$?
  if [[ ${code} -ne 0 ]]; then
    printf 'state=failed\nexit_code=%s\n' "${code}" > "${status_file}"
  fi
}
trap on_exit EXIT

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHON_BIN="${python_bin}"

for lang in "${langs[@]}"; do
  source_out=${source_root}/ua_hn/seed_${seed}/${lang}
  BASE_MODEL="${model}" TOKENIZER_NAME="${model}" CUDA_VISIBLE_DEVICES="${gpu}" \
    bash experiments/run_initial_variant.sh \
      --lang "${lang}" --seed "${seed}" --variant ua_hn --output-root "${source_root}"

  USE_LITE_LATE_INTERACTION_TRAIN=0 LI_W=0 SKIP_FINAL_RERANK=1 \
  BASE_MODEL="${model}" TOKENIZER_NAME="${model}" CUDA_VISIBLE_DEVICES="${gpu}" \
    bash experiments/run_refinement_from_initial.sh \
      --lang "${lang}" --seed "${seed}" \
      --initial-name unixcoder_ua_hn --initial-out "${source_out}" \
      --condition fc --output-root "${refinement_root}" \
      --base-model "${model}" --tokenizer-name "${model}"

  final_out=${refinement_root}/unixcoder_ua_hn/fc/seed_${seed}/${lang}
  source_eval=${evaluation_root}/${lang}/source_test
  final_eval=${evaluation_root}/${lang}/final_test
  mkdir -p "${source_eval}" "${final_eval}"

  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" experiments/run_rerank_grid.py \
    --lang "${lang}" --split test \
    --checkpoint "${source_out}/checkpoint-best-mrr/model.bin" \
    --base-model "${model}" --output-dir "${source_eval}" \
    --ks 20 --alphas 0.5 --global-only

  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" experiments/run_rerank_grid.py \
    --lang "${lang}" --split test \
    --checkpoint "${final_out}/checkpoint-best-mrr/model.bin" \
    --base-model "${model}" --output-dir "${final_eval}" \
    --ks 20 --alphas 0.5
done

"${python_bin}" experiments/summarize_unixcoder_transfer_no_li.py \
  --run-root "${run_root#./}" \
  --output-root experimental_results/12_unixcoder_transfer_no_li_seed123456

printf 'state=complete\nseed=%s\nbackbone=%s\n' "${seed}" "${model}" > "${status_file}"
trap - EXIT
