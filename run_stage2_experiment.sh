#!/usr/bin/env bash
set -euo pipefail

# Stage-2 full experiment: harvest retrieval failures from Stage-1, then train refinement.
# Default language is ruby.
# Usage:
#   bash run_stage2_experiment.sh
#   STAGE1_OUT=saved_models/refcode/stage1/ruby_seed123456_xxx bash run_stage2_experiment.sh ruby 123456

lang=${1:-ruby}
seed=${2:-123456}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
base_model=${BASE_MODEL:-DeepSoftwareAnalytics/CoCoSoDa}
tokenizer_name=${TOKENIZER_NAME:-${base_model}}

if [[ -z "${STAGE1_OUT:-}" ]]; then
  STAGE1_OUT=$(ls -td "./saved_models/refcode/stage1/${lang}_seed${seed}_"* 2>/dev/null | head -n 1 || true)
fi

if [[ -z "${STAGE1_OUT}" ]]; then
  echo "[ERROR] STAGE1_OUT is empty and no matching Stage-1 directory was found."
  echo "Set STAGE1_OUT explicitly, for example:"
  echo "STAGE1_OUT=saved_models/refcode/stage1/${lang}_seed${seed}_lr8e-6_tau0.03_relW0.25_top16_xxx bash run_stage2_experiment.sh ${lang} ${seed}"
  exit 1
fi

checkpoint_file=${LOAD_MODEL_FILE:-${STAGE1_OUT}/checkpoint-best-mrr/model.bin}
self_mined_idx_file=${SELF_MINED_IDX_FILE:-dataset/${lang}/self_mined_top32_from_stage1.pkl}

if [[ ! -f "${checkpoint_file}" ]]; then
  echo "[ERROR] Stage-1 checkpoint not found: ${checkpoint_file}"
  exit 1
fi

echo "[ReFCode-Stage2-Experiment] lang=${lang}, seed=${seed}"
echo "[ReFCode-Stage2-Experiment] base_model=${base_model}"
echo "[ReFCode-Stage2-Experiment] STAGE1_OUT=${STAGE1_OUT}"
echo "[ReFCode-Stage2-Experiment] checkpoint_file=${checkpoint_file}"
echo "[ReFCode-Stage2-Experiment] self_mined_idx_file=${self_mined_idx_file}"

python harvest_retrieval_failures.py \
  --train_data_file "dataset/${lang}/train.jsonl" \
  --output_file "${self_mined_idx_file}" \
  --model_name_or_path "${base_model}" \
  --config_name "${base_model}" \
  --tokenizer_name "${tokenizer_name}" \
  --checkpoint_file "${checkpoint_file}"

STAGE1_OUT="${STAGE1_OUT}" \
LOAD_MODEL_FILE="${checkpoint_file}" \
SELF_MINED_IDX_FILE="${self_mined_idx_file}" \
BASE_MODEL="${base_model}" \
TOKENIZER_NAME="${tokenizer_name}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
bash train_failure_calibrated_refinement.sh "${lang}" "${seed}"
