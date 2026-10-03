#!/usr/bin/env bash
# Validation-selected K/alpha effectiveness and inference-efficiency experiment.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

gpu=${CUDA_VISIBLE_DEVICES:-0}
python_bin=${PYTHON_BIN:-python}
run_root=${RUN_ROOT:-./runs/table5_efficiency_no_li}
result_root=${RESULT_ROOT:-./experimental_results/07_table5_efficiency_no_li}
base_model=${BASE_MODEL:-DeepSoftwareAnalytics/CoCoSoDa}
ks=10,20,50,100
alphas=0,0.25,0.5,0.75,1
langs=(javascript java ruby python php go)

run_grid() {
  local split=$1
  local lang=$2
  local requested_ks=$3
  local out=$4
  local checkpoint="./runs/table3_controls/components/fc_no_li/ua_hn/fc/seed_123456/${lang}/checkpoint-best-mrr/model.bin"
  local result="${out}/grid_results.json"
  if [[ -f "${result}" ]]; then
    echo "[skip] ${split}/${lang} ks=${requested_ks}: ${result}"
    return
  fi
  [[ -f "${checkpoint}" ]] || { echo "[error] missing checkpoint ${checkpoint}"; exit 1; }
  mkdir -p "${out}"
  {
    echo "language=${lang}"
    echo "split=${split}"
    echo "seed=123456"
    echo "ctrd=false"
    echo "li_auxiliary_training=false"
    echo "checkpoint=${checkpoint}"
    echo "ks=${requested_ks}"
    echo "alphas=${alphas}"
    echo "candidate_corpus=dataset/${lang}/codebase.jsonl for both valid and test queries"
    echo "latency_policy=rerank every query"
  } > "${out}/provenance.txt"
  echo "[run] ${split}/${lang} ks=${requested_ks}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u experiments/run_rerank_grid.py \
    --lang "${lang}" \
    --checkpoint "${checkpoint}" \
    --split "${split}" \
    --output-dir "${out}" \
    --ks "${requested_ks}" \
    --alphas "${alphas}" \
    --base-model "${base_model}" \
    --eval-batch-size 128 \
    --rerank-batch-size 64 \
    --code-length 256 \
    --nl-length 128 \
    --fp16 1 2>&1 | tee "${out}/running.log"
}

mkdir -p "${run_root}" "${result_root}"

echo "[phase 1/4] Validation-only K/alpha grid"
for lang in "${langs[@]}"; do
  run_grid valid "${lang}" "${ks}" "${run_root}/validation/${lang}"
done

echo "[phase 2/4] Freeze alpha for each K from validation macro MRR"
"${python_bin}" experiments/summarize_table5_efficiency.py \
  --run-root "${run_root#./}" \
  --output-root "${result_root#./}" \
  --select-only

echo "[phase 3/4] Test grid; reporting will use validation-selected alpha only"
for lang in "${langs[@]}"; do
  run_grid test "${lang}" "${ks}" "${run_root}/test/${lang}"
done

echo "[phase 4/4] Exact per-K latency/VRAM profiles on JavaScript"
for k in 10 20 50 100; do
  run_grid test javascript "${k}" "${run_root}/profile/javascript/k${k}"
done

"${python_bin}" experiments/summarize_table5_efficiency.py \
  --run-root "${run_root#./}" \
  --output-root "${result_root#./}"

echo "[done] Table 5 effectiveness-efficiency experiment completed"
