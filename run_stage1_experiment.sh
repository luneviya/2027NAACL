#!/usr/bin/env bash
set -euo pipefail

# Stage-1 full experiment: build global hard negatives, then train the initial retriever.
# Default language is ruby.
# Usage:
#   bash run_stage1_experiment.sh
#   bash run_stage1_experiment.sh python 123456

lang=${1:-ruby}
seed=${2:-123456}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
base_model=${BASE_MODEL:-DeepSoftwareAnalytics/CoCoSoDa}
tokenizer_name=${TOKENIZER_NAME:-${base_model}}

nl_length=${NL_LENGTH:-128}
encode_batch_size=${ENCODE_BATCH_SIZE:-128}
topk=${GLOBAL_HN_TOPK:-500}
bm25_rank=${GLOBAL_HN_BM25_RANK:-50}
chunk_size=${GLOBAL_HN_CHUNK_SIZE:-512}

train_file=${TRAIN_FILE:-dataset/${lang}/train.jsonl}
hard_idx_file=${HARD_IDX_FILE:-dataset/${lang}/refcode_hard_idx_top500_rank50.pkl}
embeddings_cache=${EMBEDDINGS_CACHE:-dataset/${lang}/train_query_cocosoda_emb.pt}

echo "[ReFCode-Stage1-Experiment] lang=${lang}, seed=${seed}"
echo "[ReFCode-Stage1-Experiment] base_model=${base_model}"
echo "[ReFCode-Stage1-Experiment] train_file=${train_file}"
echo "[ReFCode-Stage1-Experiment] hard_idx_file=${hard_idx_file}"
echo "[ReFCode-Stage1-Experiment] embeddings_cache=${embeddings_cache}"

python build_global_hard_negatives.py \
  --train_data_file "${train_file}" \
  --output_file "${hard_idx_file}" \
  --model_name_or_path "${base_model}" \
  --config_name "${base_model}" \
  --tokenizer_name "${tokenizer_name}" \
  --nl_length "${nl_length}" \
  --encode_batch_size "${encode_batch_size}" \
  --topk "${topk}" \
  --bm25_rank "${bm25_rank}" \
  --chunk_size "${chunk_size}" \
  --seed "${seed}" \
  --device cuda \
  --embeddings_cache "${embeddings_cache}"

HARD_IDX_FILE="${hard_idx_file}" \
BASE_MODEL="${base_model}" \
TOKENIZER_NAME="${tokenizer_name}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
bash train_initial_retriever.sh "${lang}" "${seed}"
