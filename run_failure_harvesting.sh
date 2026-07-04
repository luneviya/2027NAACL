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

if [[ -z "${INITIAL_RETRIEVER_OUT:-}" ]]; then
  INITIAL_RETRIEVER_OUT="./saved_models/initial_retriever/${lang}"
fi

if [[ -z "${INITIAL_RETRIEVER_OUT}" ]]; then
  echo "[ERROR] No retriever output directory found. Set INITIAL_RETRIEVER_OUT explicitly."
  exit 1
fi

checkpoint_file=${LOAD_MODEL_FILE:-${INITIAL_RETRIEVER_OUT}/checkpoint-best-mrr/model.bin}
self_mined_idx_file=${SELF_MINED_IDX_FILE:-dataset/${lang}/self_mined_top32_from_initial_retriever.pkl}

if [[ ! -f "${checkpoint_file}" ]]; then
  echo "[ERROR] Retriever checkpoint not found: ${checkpoint_file}"
  exit 1
fi

echo "[ReFCode-Harvesting] lang=${lang}, seed=${seed}"
echo "[ReFCode-Harvesting] base_model=${base_model}"
echo "[ReFCode-Harvesting] checkpoint_file=${checkpoint_file}"
echo "[ReFCode-Harvesting] self_mined_idx_file=${self_mined_idx_file}"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python -m refcode.run_failure_harvesting \
  --train_data_file "dataset/${lang}/train.jsonl" \
  --output_file "${self_mined_idx_file}" \
  --model_name_or_path "${base_model}" \
  --config_name "${base_model}" \
  --tokenizer_name "${tokenizer_name}" \
  --checkpoint_file "${checkpoint_file}" \
  --seed "${seed}"
