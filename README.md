# ReFCode

ReFCode is a compact research artifact for code search. The runnable pipeline is organized around the actual experiment flow: train an initial retriever, harvest retrieval failures, then run refinement with reranking and evaluation.

Large assets are intentionally excluded. Datasets, checkpoints, logs, caches, and generated indices should be prepared or produced locally.

## Repository Structure

```text
configs/                 Default experiment settings for reference
refcode/retriever/       Initial retriever training and top-K retrieval support
refcode/harvesting/      Retrieval-failure mining and data construction
refcode/refinement/      ReFCode refinement, reranking, and evaluation
refcode/utils/           Shared data, metric, parser, and I/O helpers
scripts/                 Main runnable entry scripts
```

## Environment Setup

Create an environment with either:

```bash
conda env create -f environment.yml
conda activate refcode
```

or:

```bash
pip install -r requirements.txt
```

The default encoder is loaded from Hugging Face:

```text
DeepSoftwareAnalytics/CoCoSoDa
```

## Data Preparation

Prepare CodeSearchNet-style files under:

```text
dataset/<lang>/train.jsonl
dataset/<lang>/valid.jsonl
dataset/<lang>/test.jsonl
dataset/<lang>/codebase.jsonl
```

Each example should contain code tokens or code text, natural-language tokens or text, and a stable `url` or retrieval identifier. The repository does not include raw datasets.

If the tree-sitter shared library is missing, rebuild it from the parser directory:

```bash
cd refcode/utils/parser
bash build.sh
cd -
```

## Running The Pipeline

Run commands from the repository root.

1. Train the initial retriever:

```bash
bash scripts/run_retriever.sh --lang javascript
```

2. Harvest failure candidates:

```bash
bash scripts/run_harvesting.sh --lang javascript
```

3. Run ReFCode refinement, reranking, and evaluation:

```bash
bash scripts/run_refcode.sh --lang javascript
```

Use `--seed 123456` to override the default seed. Runtime settings can also be overridden with environment variables documented in `REPRODUCIBILITY.md`.

## Expected Outputs

The pipeline writes generated artifacts locally:

```text
dataset/<lang>/refcode_hard_idx_top500_rank50.pkl
dataset/<lang>/train_query_cocosoda_emb.pt
dataset/<lang>/self_mined_top32_from_stage1.pkl
saved_models/refcode/stage1/
saved_models/refcode/stage2/
saved_models/refcode/rerank/
```

These files are ignored by git and should not be committed.

## Reproducibility Notes

Detailed commands, paths, hardware notes, and expected ruby metrics from the local verification run are in `REPRODUCIBILITY.md`.

## Anonymous Review Notes

This artifact omits author names, affiliations, acknowledgements, personal links, raw datasets, and trained checkpoints. Local paths in examples use relative paths or placeholders.
