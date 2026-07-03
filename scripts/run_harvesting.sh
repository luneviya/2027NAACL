#!/usr/bin/env bash
set -euo pipefail

lang=javascript
seed=123456

while [[ $# -gt 0 ]]; do
  case "$1" in
    --lang)
      lang="$2"
      shift 2
      ;;
    --seed)
      seed="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
base_model=${BASE_MODEL:-DeepSoftwareAnalytics/CoCoSoDa}
tokenizer_name=${TOKENIZER_NAME:-${base_model}}

if [[ -z "${STAGE1_OUT:-}" ]]; then
  STAGE1_OUT=$(ls -td "./saved_models/refcode/stage1/${lang}_seed${seed}_"* 2>/dev/null | head -n 1 || true)
fi

if [[ -z "${STAGE1_OUT}" ]]; then
  echo "[ERROR] No Stage-1 output directory found. Set STAGE1_OUT explicitly."
  exit 1
fi

checkpoint_file=${LOAD_MODEL_FILE:-${STAGE1_OUT}/checkpoint-best-mrr/model.bin}
self_mined_idx_file=${SELF_MINED_IDX_FILE:-dataset/${lang}/self_mined_top32_from_stage1.pkl}

if [[ ! -f "${checkpoint_file}" ]]; then
  echo "[ERROR] Stage-1 checkpoint not found: ${checkpoint_file}"
  exit 1
fi

echo "[ReFCode-Harvesting] lang=${lang}, seed=${seed}"
echo "[ReFCode-Harvesting] base_model=${base_model}"
echo "[ReFCode-Harvesting] checkpoint_file=${checkpoint_file}"
echo "[ReFCode-Harvesting] self_mined_idx_file=${self_mined_idx_file}"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python -m refcode.harvesting.mine \
  --train_data_file "dataset/${lang}/train.jsonl" \
  --output_file "${self_mined_idx_file}" \
  --model_name_or_path "${base_model}" \
  --config_name "${base_model}" \
  --tokenizer_name "${tokenizer_name}" \
  --checkpoint_file "${checkpoint_file}" \
  --seed "${seed}"
