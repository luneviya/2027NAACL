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

current_time=$(date "+%Y%m%d%H%M%S")

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
base_model=${BASE_MODEL:-DeepSoftwareAnalytics/CoCoSoDa}
tokenizer_name=${TOKENIZER_NAME:-${base_model}}

code_length=${CODE_LENGTH:-256}
nl_length=${NL_LENGTH:-128}
batch_size=${BATCH_SIZE:-128}
eval_batch_size=${EVAL_BATCH_SIZE:-64}

moco_k=${MOCO_K:-1024}
moco_m=${MOCO_M:-0.999}
moco_t=${MOCO_T:-0.07}

lr=${LR:-8e-6}
epoch=${EPOCH:-10}

refcode_tau=${REFCODE_TAU:-0.03}
refcode_samples=${REFCODE_SAMPLES:-4}
refcode_kl_w=${REFCODE_KL_W:-1e-5}
refcode_intra_w=${REFCODE_INTRA_W:-1.0}
refcode_hard_w=${REFCODE_HARD_W:-1.0}
refcode_hard_mode=${REFCODE_HARD_MODE:-batch_all}

ctrd_w=${CTRD_W:-0.25}
ctrd_topk=${CTRD_TOPK:-16}
ctrd_hard_w=${CTRD_HARD_W:-2.0}
ctrd_batch_w=${CTRD_BATCH_W:-1.0}
ctrd_rank_w=${CTRD_RANK_W:-0.5}
ctrd_margin=${CTRD_MARGIN:-0.2}

train_file=${TRAIN_FILE:-dataset/${lang}/train.jsonl}
hard_idx_file=${HARD_IDX_FILE:-dataset/${lang}/refcode_hard_idx_top500_rank50.pkl}
embeddings_cache=${EMBEDDINGS_CACHE:-dataset/${lang}/train_query_cocosoda_emb.pt}
output_dir=${OUTPUT_DIR:-./saved_models/initial_retriever/${lang}_seed${seed}_lr${lr}_tau${refcode_tau}_relW${ctrd_w}_top${ctrd_topk}_${current_time}}

mkdir -p "$(dirname "${hard_idx_file}")" "${output_dir}"

retrieve_args=(
  --train_data_file "${train_file}"
  --output_file "${hard_idx_file}"
  --model_name_or_path "${base_model}"
  --config_name "${base_model}"
  --tokenizer_name "${tokenizer_name}"
  --nl_length "${nl_length}"
  --encode_batch_size "${batch_size}"
  --topk "${RETRIEVAL_TOPK:-500}"
  --bm25_rank "${BM25_RANK:-50}"
  --chunk_size "${RETRIEVAL_CHUNK_SIZE:-512}"
  --seed "${seed}"
  --device "${DEVICE:-cuda}"
  --embeddings_cache "${embeddings_cache}"
)

echo "[ReFCode-Retriever] lang=${lang}, seed=${seed}"
echo "[ReFCode-Retriever] base_model=${base_model}"
echo "[ReFCode-Retriever] train_file=${train_file}"
echo "[ReFCode-Retriever] hard_idx_file=${hard_idx_file}"
echo "[ReFCode-Retriever] embeddings_cache=${embeddings_cache}"
echo "[ReFCode-Retriever] output_dir=${output_dir}"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python refcode/initial_retriever/run.py "${retrieve_args[@]}"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python refcode/refcode_refinement/run.py \
  --eval_frequency 100 \
  --moco_m ${moco_m} \
  --moco_t ${moco_t} \
  --model_type base \
  --output_dir ${output_dir} \
  --data_aug_type random_mask \
  --moco_k ${moco_k} \
  --config_name=${base_model} \
  --model_name_or_path=${base_model} \
  --tokenizer_name=${tokenizer_name} \
  --lang=${lang} \
  --do_train \
  --do_test \
  --train_data_file=${train_file} \
  --eval_data_file=dataset/${lang}/valid.jsonl \
  --test_data_file=dataset/${lang}/test.jsonl \
  --codebase_file=dataset/${lang}/codebase.jsonl \
  --num_train_epochs ${epoch} \
  --code_length ${code_length} \
  --nl_length ${nl_length} \
  --train_batch_size ${batch_size} \
  --eval_batch_size ${eval_batch_size} \
  --learning_rate ${lr} \
  --seed ${seed} \
  --use_refcode_uncertainty \
  --refcode_temperature ${refcode_tau} \
  --refcode_uncertainty_samples ${refcode_samples} \
  --refcode_intra_weight ${refcode_intra_w} \
  --refcode_kl_weight ${refcode_kl_w} \
  --use_refcode_global_hard_negative \
  --refcode_hard_idx_file ${hard_idx_file} \
  --refcode_hard_mode ${refcode_hard_mode} \
  --refcode_hard_weight ${refcode_hard_w} \
  --use_ctrd \
  --ctrd_use_offline_hard \
  --ctrd_weight ${ctrd_w} \
  --ctrd_batch_topk ${ctrd_topk} \
  --ctrd_hard_weight ${ctrd_hard_w} \
  --ctrd_batch_weight ${ctrd_batch_w} \
  --ctrd_rank_weight ${ctrd_rank_w} \
  --ctrd_rank_margin ${ctrd_margin} \
  2>&1 | tee ${output_dir}/running.log
