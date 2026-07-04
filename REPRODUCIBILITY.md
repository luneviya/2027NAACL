# Reproducibility

This file records the commands and assumptions needed to reproduce the main ReFCode pipeline. The repository contains runnable code only; datasets, checkpoints, generated indices, logs, and caches are intentionally excluded.

## Dataset Paths

Place CodeSearchNet-style data under:

```text
dataset/<lang>/train.jsonl
dataset/<lang>/valid.jsonl
dataset/<lang>/test.jsonl
dataset/<lang>/codebase.jsonl
```

For example:

```text
dataset/ruby/train.jsonl
dataset/ruby/valid.jsonl
dataset/ruby/test.jsonl
dataset/ruby/codebase.jsonl
```

The scripts use relative paths and should be run from the repository root.

## Output Paths

Generated files are written to:

```text
dataset/<lang>/refcode_hard_idx_top500_rank50.pkl
dataset/<lang>/train_query_cocosoda_emb.pt
dataset/<lang>/self_mined_top32_from_initial_retriever.pkl
saved_models/initial_retriever/
saved_models/refcode/
```

Override output locations with:

```bash
HARD_IDX_FILE=/path/to/hard.pkl
SELF_MINED_IDX_FILE=/path/to/self_mined.pkl
OUTPUT_DIR=/path/to/output
```

## Hardware And Software

The local verification used one CUDA GPU. Full experiments are GPU-oriented because they encode the full training and codebase splits and train Transformer encoders.

Core dependencies are listed in:

```text
requirements.txt
environment.yml
```

The default pretrained encoder is:

```text
DeepSoftwareAnalytics/CoCoSoDa
```

Override it with:

```bash
BASE_MODEL=/path/to/model-or-huggingface-id
TOKENIZER_NAME=/path/to/tokenizer-or-huggingface-id
```

## Random Seeds

The default seed is:

```text
123456
```

Use:

```bash
bash scripts/run_initial_retriever.sh --lang ruby --seed 123456
bash scripts/run_failure_harvesting.sh --lang ruby --seed 123456
bash scripts/run_refcode.sh --lang ruby --seed 123456
```

## Main Commands

The Initial Retriever step trains the initial retriever. It first builds the global hard-negative index, then trains with the same default hyperparameters used by the local verification run.

```bash
bash scripts/run_initial_retriever.sh --lang ruby
```

During hard-negative construction, the script now writes or reuses:

```text
dataset/<lang>/train_query_cocosoda_emb.pt
```

Override this cache path with:

```bash
EMBEDDINGS_CACHE=/path/to/train_query_embeddings.pt bash scripts/run_initial_retriever.sh --lang ruby
```

The Failure Harvesting step constructs retrieval failures from the best initial retriever checkpoint.

```bash
bash scripts/run_failure_harvesting.sh --lang ruby
```

The ReFCode Refinement step trains from the initial retriever checkpoint with harvested failures, then runs final reranking.

```bash
bash scripts/run_refcode.sh --lang ruby
```

## Important Defaults

Initial Retriever:

```text
learning_rate=8e-6
num_train_epochs=10
train_batch_size=128
nl_length=128
code_length=256
seed=123456
global_hard_topk=500
bm25_rank=50
refcode_temperature=0.03
ctrd_weight=0.25
ctrd_batch_topk=16
```

ReFCode Refinement:

```text
learning_rate=5e-6
num_train_epochs=3
train_batch_size=128
self_mined_weight=0.45
self_mined_topk=8
self_mined_train_k=1
li_weight=0.05
valid_fusion_topk=50
valid_fusion_alpha=0.5
```

## Expected Metrics From Local Ruby Verification

The following values were observed locally for ruby with seed `123456` and the default Hugging Face encoder. They are reported as verification context, not as a newly rerun claim after this packaging change.

Initial Retriever test:

```text
R@1 = 0.730
R@5 = 0.948
R@10 = 0.971
MRR = 0.8239225133545454
```

ReFCode Refinement test:

```text
R@1 = 0.745
R@5 = 0.946
R@10 = 0.972
MRR = 0.8321392022782704
```

## Smoke Check

The lightest repository check does not run training. It verifies shell syntax and Python import/compile paths:

```bash
bash -n scripts/run_initial_retriever.sh scripts/run_failure_harvesting.sh scripts/run_refcode.sh refcode/utils/parser/build.sh
```

```bash
python -m py_compile \
  refcode/refcode_refinement/model.py \
  refcode/refcode_refinement/run.py \
  refcode/refcode_refinement/rerank.py \
  refcode/initial_retriever/run.py \
  refcode/failure_harvesting/run.py \
  refcode/utils/utils.py \
  refcode/utils/data_utils.py \
  refcode/utils/metrics.py \
  refcode/utils/parser/DFG.py \
  refcode/utils/parser/build.py \
  refcode/utils/parser/utils.py
```

For a functional small-data smoke test, prepare a tiny CodeSearchNet-style dataset and run the three main scripts with small environment overrides, for example:

```bash
RETRIEVAL_TOPK=8 BM25_RANK=2 BATCH_SIZE=4 EPOCH=1 bash scripts/run_initial_retriever.sh --lang ruby
bash scripts/run_failure_harvesting.sh --lang ruby
BATCH_SIZE=4 EPOCH=1 bash scripts/run_refcode.sh --lang ruby
```

The tiny-data smoke test requires a valid dataset and encoder checkpoint.
