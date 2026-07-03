# ReFCode

This repository contains the compact main-experiment implementation for ReFCode on CodeSearchNet-style code search.

The released pipeline keeps only the core paper workflow:

1. Build global hard negatives.
2. Run Stage-1 ReFCode training with relevance refinement.
3. Build self-mined retrieval-failure negatives from the Stage-1 checkpoint.
4. Run Stage-2 refinement with self-mined failures and lightweight late interaction.
5. Run final late-interaction reranking.

Large assets are intentionally excluded, including datasets, checkpoints, pretrained backbones, logs, caches, and intermediate index files.

## Files

Core Python files:

- `model.py`: ReFCode encoder, uncertainty heads, and relevance head.
- `run.py`: unified Stage-1/Stage-2 training and evaluation entry point.
- `build_global_hard_negatives.py`: constructs Stage-1 global hard negatives.
- `harvest_retrieval_failures.py`: constructs Stage-2 self-mined retrieval failures.
- `run_dual_granularity_rerank.py`: final dual-granularity late-interaction reranking.
- `utils.py`: minimal JSON/pickle persistence helpers.
- `parser/`: tree-sitter data-flow parser utilities.

Main scripts:

- `train_initial_retriever.sh`
- `train_failure_calibrated_refinement.sh`
- `run_stage1_experiment.sh`
- `run_stage2_experiment.sh`
- `run_rerank.sh`

## Expected Local Assets

Place runtime assets locally when running experiments:

- `dataset/<lang>/{train,valid,test,codebase}.jsonl`
- a compatible encoder checkpoint such as `DeepSoftwareAnalytics/CoCoSoDa`, or a local path supplied through `BASE_MODEL`
- generated hard-negative files under `dataset/<lang>/`
- checkpoints under `saved_models/`

These paths are ignored by `.gitignore` and should not be committed.

## Workflow

Default Ruby two-stage experiments:

```bash
bash run_stage1_experiment.sh
```

```bash
bash run_stage2_experiment.sh
```

Use another language by passing language and seed:

```bash
bash run_stage1_experiment.sh python 123456
bash run_stage2_experiment.sh python 123456
```

Build global hard negatives:

```bash
python build_global_hard_negatives.py \
  --train_data_file dataset/java/train.jsonl \
  --output_file dataset/java/refcode_hard_idx_top500_rank50.pkl \
  --model_name_or_path DeepSoftwareAnalytics/CoCoSoDa \
  --config_name DeepSoftwareAnalytics/CoCoSoDa \
  --tokenizer_name DeepSoftwareAnalytics/CoCoSoDa
```

Run Stage-1:

```bash
HARD_IDX_FILE=dataset/java/refcode_hard_idx_top500_rank50.pkl \
bash train_initial_retriever.sh java 123456
```

Build self-mined retrieval failures:

```bash
python harvest_retrieval_failures.py \
  --train_data_file dataset/java/train.jsonl \
  --output_file dataset/java/self_mined_top32_from_stage1.pkl \
  --model_name_or_path DeepSoftwareAnalytics/CoCoSoDa \
  --config_name DeepSoftwareAnalytics/CoCoSoDa \
  --tokenizer_name DeepSoftwareAnalytics/CoCoSoDa \
  --loaded_model_filename saved_models/refcode/stage1/<run>/checkpoint-best-mrr/model.bin
```

Run Stage-2:

```bash
STAGE1_OUT=saved_models/refcode/stage1/<run> \
SELF_MINED_IDX_FILE=dataset/java/self_mined_top32_from_stage1.pkl \
bash train_failure_calibrated_refinement.sh java 123456
```

Run final reranking:

```bash
bash run_rerank.sh \
  java \
  saved_models/refcode/stage2/<run>/checkpoint-best-mrr/model.bin
```
