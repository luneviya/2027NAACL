#!/usr/bin/env bash
set -euo pipefail

# Final ReFCode late-interaction reranking for a Stage-2 checkpoint.
# Usage:
#   bash run_rerank.sh java saved_models/refcode/stage2/<run>/checkpoint-best-mrr/model.bin

lang=${1:-java}
checkpoint=${2:-}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
base_model=${BASE_MODEL:-DeepSoftwareAnalytics/CoCoSoDa}
tokenizer_name=${TOKENIZER_NAME:-${base_model}}

code_length=${CODE_LENGTH:-256}
nl_length=${NL_LENGTH:-128}
eval_batch_size=${EVAL_BATCH_SIZE:-128}
rerank_batch_size=${RERANK_BATCH_SIZE:-64}
top_k=${TOP_K:-50}
fusion_alpha=${FUSION_ALPHA:-0.5}
fp16=${FP16:-1}

if [[ -z "${checkpoint}" ]]; then
  checkpoint=$(ls -td ./saved_models/refcode/stage2/${lang}_*/checkpoint-best-mrr/model.bin 2>/dev/null | head -n 1 || true)
fi

if [[ -z "${checkpoint}" || ! -f "${checkpoint}" ]]; then
  echo "[ERROR] Stage-2 checkpoint not found. Pass it as the second argument."
  exit 1
fi

output_dir=${OUTPUT_DIR:-./saved_models/refcode/rerank/${lang}_top${top_k}_alpha${fusion_alpha}}
mkdir -p "${output_dir}"

echo "[ReFCode-Rerank] lang=${lang}"
echo "[ReFCode-Rerank] checkpoint=${checkpoint}"
echo "[ReFCode-Rerank] output_dir=${output_dir}"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python run_dual_granularity_rerank.py \
  --lang ${lang} \
  --model_name_or_path ${base_model} \
  --config_name ${base_model} \
  --tokenizer_name ${tokenizer_name} \
  --loaded_model_filename ${checkpoint} \
  --eval_data_file dataset/${lang}/test.jsonl \
  --codebase_file dataset/${lang}/codebase.jsonl \
  --output_dir ${output_dir} \
  --code_length ${code_length} \
  --nl_length ${nl_length} \
  --eval_batch_size ${eval_batch_size} \
  --rerank_batch_size ${rerank_batch_size} \
  --top_k ${top_k} \
  --fusion_alpha ${fusion_alpha} \
  --fp16 ${fp16} \
  2>&1 | tee ${output_dir}/running.log
