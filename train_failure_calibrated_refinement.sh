#!/usr/bin/env bash
set -euo pipefail

# Stage-2 ReFCode refinement with self-mined failures, lightweight late interaction,
# and validation-fusion checkpoint selection.
# Usage:
#   STAGE1_OUT=saved_models/refcode/stage1/<run> SELF_MINED_IDX_FILE=dataset/java/self_mined_top32_from_stage1.pkl bash train_failure_calibrated_refinement.sh java 123456

lang=${1:-java}
seed=${2:-123456}
current_time=$(date "+%Y%m%d%H%M%S")

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:32}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
base_model=${BASE_MODEL:-DeepSoftwareAnalytics/CoCoSoDa}
tokenizer_name=${TOKENIZER_NAME:-${base_model}}

code_length=${CODE_LENGTH:-256}
nl_length=${NL_LENGTH:-128}

lr=${LR:-5e-6}
epoch=${EPOCH:-3}
batch_size=${BATCH_SIZE:-128}
eval_batch_size=${EVAL_BATCH_SIZE:-64}

moco_k=${MOCO_K:-1024}
moco_m=${MOCO_M:-0.999}
moco_t=${MOCO_T:-0.07}

refcode_tau=${REFCODE_TAU:-0.03}
refcode_samples=${REFCODE_SAMPLES:-1}
refcode_intra_w=${REFCODE_INTRA_W:-1.0}
refcode_kl_w=${REFCODE_KL_W:-1e-5}

self_mine_w=${SELF_MINE_W:-0.45}
self_mine_topk=${SELF_MINE_TOPK:-8}
self_mine_train_k=${SELF_MINE_TRAIN_K:-1}
self_mine_temp=${SELF_MINE_TEMP:-0.03}
self_mine_margin=${SELF_MINE_MARGIN:-0.02}
self_mine_max_w=${SELF_MINE_MAX_W:-0}

use_lite_late_interaction_train=${USE_LITE_LATE_INTERACTION_TRAIN:-1}
li_weight=${LI_W:-0.05}
li_temperature=${LI_TEMP:-0.05}
li_train_k=${LI_TRAIN_K:-1}
li_sample_size=${LI_SAMPLE_SIZE:-16}
li_every_n_steps=${LI_EVERY_N_STEPS:-2}
li_chunk_size=${LI_CHUNK_SIZE:-16}
li_nl_length=${LI_NL_LENGTH:-64}
li_code_length=${LI_CODE_LENGTH:-128}

use_valid_fusion_select=${USE_VALID_FUSION_SELECT:-1}
valid_fusion_topk=${VALID_FUSION_TOPK:-50}
valid_fusion_alpha=${VALID_FUSION_ALPHA:-0.5}
valid_rerank_batch_size=${VALID_RERANK_BATCH_SIZE:-64}
valid_fusion_fp16=${VALID_FUSION_FP16:-1}

if [[ -z "${STAGE1_OUT:-}" ]]; then
  STAGE1_OUT=$(ls -td ./saved_models/refcode/stage1/${lang}_seed${seed}_* 2>/dev/null | head -n 1 || true)
fi

if [[ -z "${STAGE1_OUT}" ]]; then
  echo "[ERROR] STAGE1_OUT is empty and no matching Stage-1 directory was found."
  echo "Set STAGE1_OUT to a Stage-1 output directory, for example:"
  echo "STAGE1_OUT=./saved_models/refcode/stage1/java_seed123456_lr8e-6_tau0.03_relW0.25_top16_xxx"
  exit 1
fi

load_model_file=${LOAD_MODEL_FILE:-${STAGE1_OUT}/checkpoint-best-mrr/model.bin}
self_mined_idx_file=${SELF_MINED_IDX_FILE:-dataset/${lang}/self_mined_top32_from_stage1.pkl}
output_dir=${OUTPUT_DIR:-./saved_models/refcode/stage2/${lang}_seed${seed}_K${self_mine_train_k}_W${self_mine_w}_LI${li_weight}_${current_time}}

if [[ ! -f "${load_model_file}" ]]; then
  echo "[ERROR] Stage-1 checkpoint not found: ${load_model_file}"
  exit 1
fi

if [[ ! -f "${self_mined_idx_file}" ]]; then
  echo "[ERROR] Self-mined failure file not found: ${self_mined_idx_file}"
  echo "Build it first with harvest_retrieval_failures.py."
  exit 1
fi

mkdir -p "${output_dir}"

echo "[ReFCode-Stage2] lang=${lang}, seed=${seed}"
echo "[ReFCode-Stage2] base_model=${base_model}"
echo "[ReFCode-Stage2] STAGE1_OUT=${STAGE1_OUT}"
echo "[ReFCode-Stage2] load_model_file=${load_model_file}"
echo "[ReFCode-Stage2] self_mined_idx_file=${self_mined_idx_file}"
echo "[ReFCode-Stage2] output_dir=${output_dir}"
echo "[ReFCode-Stage2] lr=${lr}, epoch=${epoch}, batch_size=${batch_size}, samples=${refcode_samples}"
echo "[ReFCode-Stage2] self_mine_w=${self_mine_w}, topk=${self_mine_topk}, train_k=${self_mine_train_k}"
echo "[ReFCode-Stage2] late_interaction=${use_lite_late_interaction_train}, li_w=${li_weight}, li_temp=${li_temperature}, li_train_k=${li_train_k}"
echo "[ReFCode-Stage2] valid_fusion_select=${use_valid_fusion_select}, topk=${valid_fusion_topk}, alpha=${valid_fusion_alpha}"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python run.py \
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
  --loaded_model_filename ${load_model_file} \
  --lang=${lang} \
  --do_train \
  --do_test \
  --train_data_file=dataset/${lang}/train.jsonl \
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
  --use_self_mined_hard_negative \
  --self_mined_idx_file ${self_mined_idx_file} \
  --self_mined_weight ${self_mine_w} \
  --self_mined_topk ${self_mine_topk} \
  --self_mined_train_k ${self_mine_train_k} \
  --self_mined_temperature ${self_mine_temp} \
  --self_mined_margin ${self_mine_margin} \
  --self_mined_max_weight ${self_mine_max_w} \
  --use_lite_late_interaction_train ${use_lite_late_interaction_train} \
  --li_weight ${li_weight} \
  --li_temperature ${li_temperature} \
  --li_train_k ${li_train_k} \
  --li_sample_size ${li_sample_size} \
  --li_every_n_steps ${li_every_n_steps} \
  --li_chunk_size ${li_chunk_size} \
  --li_nl_length ${li_nl_length} \
  --li_code_length ${li_code_length} \
  --use_valid_fusion_select ${use_valid_fusion_select} \
  --valid_fusion_topk ${valid_fusion_topk} \
  --valid_fusion_alpha ${valid_fusion_alpha} \
  --valid_rerank_batch_size ${valid_rerank_batch_size} \
  --valid_fusion_fp16 ${valid_fusion_fp16} \
  2>&1 | tee ${output_dir}/running.log
