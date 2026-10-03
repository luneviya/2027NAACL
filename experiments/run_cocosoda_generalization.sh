#!/usr/bin/env bash
# Pure UniXcoder-family backbone -> ReFCode-G/F on all six CodeSearchNet languages.
#
# The local source stage exists only to create the frozen checkpoint required
# for failure harvesting. It is validation-selected but never evaluated on the
# test set or reported as a reproduced baseline. Paper source values remain the
# values reported by the original CoCoSoDa paper.
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${repo}"

cocosoda_repo=${COCOSODA_REPO:-${repo}/../CoCoSoDa}
python_bin=${PYTHON_BIN:-python}
model=${BACKBONE_MODEL:-${COCOSODA_MODEL:-DeepSoftwareAnalytics/CoCoSoDa}}
backbone_name=${BACKBONE_NAME:-cocosoda}
source_name=${SOURCE_NAME:-pure_${backbone_name}}
gpu=${CUDA_VISIBLE_DEVICES:-0}
seed=123456
run_root=${BACKBONE_GENERALIZATION_ROOT:-${COCOSODA_GENERALIZATION_ROOT:-./runs/backbone_generalization_no_ua/${backbone_name}}}
if [[ "${run_root}" != /* ]]; then
  run_root=${repo}/${run_root#./}
fi
source_root=${run_root}/source/seed_${seed}
candidate_root=${run_root}/candidate_pools/seed_${seed}
refinement_root=${run_root}/refinement
evaluation_root=${run_root}/evaluation
timing_file=${run_root}/timings.tsv
timing_summary=${run_root}/TIMING_SUMMARY.txt
status_file=${run_root}/STATUS.txt
langs=(java javascript ruby python php go)

[[ -x "${python_bin}" ]] || { echo "Missing Python: ${python_bin}" >&2; exit 1; }
if [[ -d "${model}" && ( ! -f "${model}/config.json" || ! -f "${model}/pytorch_model.bin" ) ]]; then
  echo "Incomplete local backbone model: ${model}" >&2
  exit 1
fi
[[ -f "${cocosoda_repo}/run.py" ]] || { echo "Missing CoCoSoDa run.py" >&2; exit 1; }

mkdir -p "${run_root}" "${source_root}" "${candidate_root}" "${evaluation_root}"
if [[ ! -f "${timing_file}" ]]; then
  printf 'language\tstage\tstarted_at\tended_at\telapsed_seconds\tstatus\n' > "${timing_file}"
fi

overall_started_epoch=$(date +%s)
overall_started_at=$(date --iso-8601=seconds)
printf 'state=running\nseed=%s\nbackbone_name=%s\nbackbone_model=%s\noverall_started_at=%s\n' \
  "${seed}" "${backbone_name}" "${model}" "${overall_started_at}" > "${status_file}"

write_status() {
  local state=$1
  local lang=$2
  local stage=$3
  printf 'state=%s\nseed=%s\nbackbone_name=%s\nbackbone_model=%s\ncurrent_language=%s\ncurrent_stage=%s\noverall_started_at=%s\n' \
    "${state}" "${seed}" "${backbone_name}" "${model}" "${lang}" "${stage}" "${overall_started_at}" > "${status_file}"
}

run_stage() {
  local lang=$1
  local stage=$2
  local log_file=$3
  shift 3
  local started_epoch started_at ended_epoch ended_at elapsed exit_code

  mkdir -p "$(dirname "${log_file}")"
  write_status running "${lang}" "${stage}"
  started_epoch=$(date +%s)
  started_at=$(date --iso-8601=seconds)
  echo "[timing] start language=${lang} stage=${stage} at=${started_at}"

  set +e
  "$@" 2>&1 | tee "${log_file}"
  exit_code=${PIPESTATUS[0]}
  set -e

  ended_epoch=$(date +%s)
  ended_at=$(date --iso-8601=seconds)
  elapsed=$((ended_epoch - started_epoch))
  if [[ ${exit_code} -eq 0 ]]; then
    printf '%s\t%s\t%s\t%s\t%s\tsuccess\n' \
      "${lang}" "${stage}" "${started_at}" "${ended_at}" "${elapsed}" >> "${timing_file}"
    echo "[timing] done language=${lang} stage=${stage} elapsed_seconds=${elapsed}"
  else
    printf '%s\t%s\t%s\t%s\t%s\tfailed\n' \
      "${lang}" "${stage}" "${started_at}" "${ended_at}" "${elapsed}" >> "${timing_file}"
    write_status failed "${lang}" "${stage}"
    echo "[error] language=${lang} stage=${stage} exit_code=${exit_code}" >&2
    exit "${exit_code}"
  fi
}

prepare_source() {
  local lang=$1
  local out=$2
  local data_dir=${cocosoda_repo}/dataset/${lang}
  mkdir -p "${out}"
  {
    echo "role=internal_source_checkpoint_only"
    echo "paper_source_score=reported_from_original_paper"
    echo "test_evaluation=false"
    echo "method=${source_name}"
    echo "backbone_name=${backbone_name}"
    echo "pretrained_model=${model}"
    echo "language=${lang}"
    echo "seed=${seed}"
    echo "epochs=5"
    echo "learning_rate=2e-5"
    echo "train_batch_size=128"
  } > "${out}/provenance.txt"
  (
    cd "${cocosoda_repo}"
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=${seed} \
      CUDA_VISIBLE_DEVICES=${gpu} "${python_bin}" run.py \
        --eval_frequency 100 \
        --moco_m 0.999 \
        --moco_t 0.07 \
        --model_type base \
        --output_dir "${out}" \
        --data_aug_type random_mask \
        --moco_k 1024 \
        --config_name "${model}" \
        --model_name_or_path "${model}" \
        --tokenizer_name "${model}" \
        --lang "${lang}" \
        --do_train \
        --train_data_file "${data_dir}/train.jsonl" \
        --eval_data_file "${data_dir}/valid.jsonl" \
        --test_data_file "${data_dir}/test.jsonl" \
        --codebase_file "${data_dir}/codebase.jsonl" \
        --num_train_epochs 5 \
        --code_length 256 \
        --nl_length 128 \
        --train_batch_size 128 \
        --eval_batch_size 64 \
        --learning_rate 2e-5 \
        --seed ${seed}
  )
}

mine_candidates() {
  local lang=$1
  local source_out=$2
  local candidate_file=$3
  CUDA_VISIBLE_DEVICES=${gpu} "${python_bin}" -m refcode.run_failure_harvesting \
    --train_data_file "dataset/${lang}/train.jsonl" \
    --output_file "${candidate_file}" \
    --model_name_or_path "${model}" \
    --config_name "${model}" \
    --tokenizer_name "${model}" \
    --checkpoint_file "${source_out}/checkpoint-best-mrr/model.bin" \
    --topk 32 \
    --seed ${seed}
}

train_refcode_g() {
  local lang=$1
  local source_out=$2
  local candidate_file=$3
  USE_REFCODE_UNCERTAINTY=0 \
  USE_LITE_LATE_INTERACTION_TRAIN=0 \
  LI_W=0 \
  USE_RELIABLE_LOCAL_DISTILLATION=0 \
  USE_VALID_FUSION_SELECT=1 \
  VALID_FUSION_TOPK=20 \
  VALID_FUSION_ALPHA=0.5 \
  SKIP_FINAL_RERANK=1 \
  BASE_MODEL="${model}" \
  TOKENIZER_NAME="${model}" \
  PYTHON_BIN="${python_bin}" \
  CUDA_VISIBLE_DEVICES=${gpu} \
    bash experiments/run_refinement_from_initial.sh \
      --lang "${lang}" \
      --seed "${seed}" \
      --initial-name "${source_name}" \
      --initial-out "${source_out}" \
      --condition fc \
      --candidate-file "${candidate_file}" \
      --output-root "${refinement_root}" \
      --base-model "${model}" \
      --tokenizer-name "${model}"
}

evaluate_refcode_gf() {
  local lang=$1
  local checkpoint=$2
  local out=$3
  CUDA_VISIBLE_DEVICES=${gpu} "${python_bin}" -u experiments/run_rerank_grid.py \
    --lang "${lang}" \
    --checkpoint "${checkpoint}" \
    --split test \
    --output-dir "${out}" \
    --ks 20 \
    --alphas 0.5 \
    --base-model "${model}" \
    --eval-batch-size 128 \
    --rerank-batch-size 64 \
    --code-length 256 \
    --nl-length 128 \
    --fp16 1
}

for lang in "${langs[@]}"; do
  source_out=${source_root}/${lang}
  source_checkpoint=${source_out}/checkpoint-best-mrr/model.bin
  candidate_file=${candidate_root}/${lang}_fc_top32.pkl
  refinement_out=${refinement_root}/${source_name}/fc/seed_${seed}/${lang}
  refinement_checkpoint=${refinement_out}/checkpoint-best-mrr/model.bin
  evaluation_out=${evaluation_root}/${lang}

  if [[ ! -f "${source_checkpoint}" ]]; then
    run_stage "${lang}" source_prepare "${source_out}/source_training.log" \
      prepare_source "${lang}" "${source_out}"
  else
    echo "[skip] source checkpoint exists: ${source_checkpoint}"
  fi

  if [[ ! -f "${candidate_file}" ]]; then
    run_stage "${lang}" candidate_mining "${candidate_root}/${lang}_mining.log" \
      mine_candidates "${lang}" "${source_out}" "${candidate_file}"
  else
    echo "[skip] candidate cache exists: ${candidate_file}"
  fi

  if [[ ! -f "${refinement_checkpoint}" ]]; then
    run_stage "${lang}" refcode_g_train "${refinement_out}/timed_training.log" \
      train_refcode_g "${lang}" "${source_out}" "${candidate_file}"
  else
    echo "[skip] ReFCode-G checkpoint exists: ${refinement_checkpoint}"
  fi

  if [[ ! -f "${evaluation_out}/grid_results.json" ]]; then
    run_stage "${lang}" refcode_gf_eval "${evaluation_out}/timed_evaluation.log" \
      evaluate_refcode_gf "${lang}" "${refinement_checkpoint}" "${evaluation_out}"
  else
    echo "[skip] G/F evaluation exists: ${evaluation_out}/grid_results.json"
  fi
done

overall_ended_epoch=$(date +%s)
overall_ended_at=$(date --iso-8601=seconds)
overall_elapsed=$((overall_ended_epoch - overall_started_epoch))

awk -F '\t' '
  NR > 1 && $6 == "success" { stage[$2] += $5; success += $5 }
  NR > 1 { attempted += $5 }
  END {
    print "successful_stage_gpu_seconds=" success
    print "attempted_gpu_seconds=" attempted
    for (name in stage) print name "_seconds=" stage[name]
  }
' "${timing_file}" | sort > "${timing_summary}"
printf 'overall_wall_seconds=%s\noverall_started_at=%s\noverall_ended_at=%s\n' \
  "${overall_elapsed}" "${overall_started_at}" "${overall_ended_at}" >> "${timing_summary}"

printf 'state=complete\nseed=%s\nbackbone_name=%s\nbackbone_model=%s\noverall_started_at=%s\noverall_ended_at=%s\noverall_wall_seconds=%s\n' \
  "${seed}" "${backbone_name}" "${model}" "${overall_started_at}" "${overall_ended_at}" "${overall_elapsed}" > "${status_file}"
echo "[done] ${source_name} ReFCode-G/F six-language generalization completed"
