#!/usr/bin/env bash
# Reproduce the official CoCoSoDa fine-tuning results without modifying the
# original repository. The defaults mirror CoCoSoDa's committed
# run_fine_tune.sh (5 epochs, lr=2e-5, batch=128, seed=123456).

set -euo pipefail

artifact_repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
repo=${COCOSODA_REPO:-${artifact_repo}/../CoCoSoDa}
python_bin=${PYTHON_BIN:-python}
model_path=${COCOSODA_MODEL:-DeepSoftwareAnalytics/CoCoSoDa}
output_root=${artifact_repo}/experimental_results/06_reproduced_baselines/cocosoda/seed_123456
gpu=0
langs=(java javascript ruby python php go)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) gpu="$2"; shift 2 ;;
    --langs) IFS=',' read -r -a langs <<< "$2"; shift 2 ;;
    --output-root) output_root="$2"; shift 2 ;;
    --model-path) model_path="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ -x "$python_bin" ]] || { echo "Missing Python: $python_bin" >&2; exit 1; }
[[ -f "$repo/run.py" ]] || { echo "Missing CoCoSoDa runner: $repo/run.py" >&2; exit 1; }
if [[ -d "$model_path" && ! -f "$model_path/pytorch_model.bin" ]]; then
  echo "Incomplete local CoCoSoDa checkpoint: $model_path" >&2
  exit 1
fi

mkdir -p "$output_root"

for lang in "${langs[@]}"; do
  data_dir="$repo/dataset/$lang"
  run_dir="$output_root/$lang"
  checkpoint="$run_dir/checkpoint-best-mrr/model.bin"
  result="$run_dir/result.jsonl"

  for split in train valid test codebase; do
    [[ -f "$data_dir/$split.jsonl" ]] || {
      echo "Missing $lang data: $data_dir/$split.jsonl" >&2
      exit 1
    }
  done

  if [[ -f "$checkpoint" && -f "$result" ]]; then
    echo "[skip] Completed CoCoSoDa reproduction: $lang"
    continue
  fi

  mkdir -p "$run_dir"
  {
    echo "method=CoCoSoDa"
    echo "implementation=$repo"
    echo "implementation_commit=$(git -C "$repo" rev-parse HEAD)"
    echo "pretrained_model=$model_path"
    echo "language=$lang"
    echo "seed=123456"
    echo "epochs=5"
    echo "learning_rate=2e-5"
    echo "train_batch_size=128"
    echo "eval_batch_size=64"
    echo "code_length=256"
    echo "nl_length=128"
    echo "data_aug_type=random_mask"
    sha256sum "$data_dir/train.jsonl" "$data_dir/valid.jsonl" \
      "$data_dir/test.jsonl" "$data_dir/codebase.jsonl"
  } > "$run_dir/provenance.txt"

  mode_args=(--do_train --do_test)
  log_file="$run_dir/running.log"
  if [[ -f "$checkpoint" ]]; then
    mode_args=(--do_test)
    log_file="$run_dir/resume_evaluation.log"
    echo "[resume] Checkpoint found; evaluating $lang"
  else
    echo "[run] Training and evaluating CoCoSoDa: $lang"
  fi

  (
    cd "$repo"
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=123456 \
      CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" run.py \
        --eval_frequency 100 \
        --moco_m 0.999 \
        --moco_t 0.07 \
        --model_type base \
        --output_dir "$run_dir" \
        --data_aug_type random_mask \
        --moco_k 1024 \
        --config_name "$model_path" \
        --model_name_or_path "$model_path" \
        --tokenizer_name "$model_path" \
        --lang "$lang" \
        "${mode_args[@]}" \
        --train_data_file "$data_dir/train.jsonl" \
        --eval_data_file "$data_dir/valid.jsonl" \
        --test_data_file "$data_dir/test.jsonl" \
        --codebase_file "$data_dir/codebase.jsonl" \
        --num_train_epochs 5 \
        --code_length 256 \
        --nl_length 128 \
        --train_batch_size 128 \
        --eval_batch_size 64 \
        --learning_rate 2e-5 \
        --seed 123456 2>&1 | tee "$log_file"
  )
done

echo "[done] CoCoSoDa reproduction: $output_root"
