# Reviewer Release Checklist

This checklist records the current artifact-readiness review for the ReFCode repository. It is intentionally diagnostic only: no algorithm logic, experiment outputs, or existing source files were changed during this review.

## Current Status

- Status: not yet ready for reviewer/GitHub release without cleanup.
- Main reason: the working tree currently contains generated runtime artifacts (`saved_models/`, checkpoints, logs, `__pycache__/`) and a `dataset` symlink that points to the old local CoCoSoDa directory.
- Core runnable code is present for the main pipeline, but packaging and reviewer instructions need a small pass before release.

## 1. Necessary Files Only

Current core files that appear necessary for reviewer evaluation:

- `model.py`
- `run.py`
- `utils.py`
- `build_global_hard_negatives.py`
- `harvest_retrieval_failures.py`
- `run_dual_granularity_rerank.py`
- `train_initial_retriever.sh`
- `train_failure_calibrated_refinement.sh`
- `run_stage1_experiment.sh`
- `run_stage2_experiment.sh`
- `run_rerank.sh`
- `parser/DFG.py`
- `parser/__init__.py`
- `parser/build.py`
- `parser/build.sh`
- `parser/utils.py`
- `parser/my-languages.so`
- `README.md`
- `.gitignore`

Files/directories currently present but should not be committed:

- `saved_models/`
- `saved_models/refcode/stage1/.../checkpoint-best-mrr/model.bin`
- `saved_models/refcode/stage2/.../checkpoint-best-mrr/model.bin`
- `saved_models/refcode/stage1/.../running.log`
- `saved_models/refcode/stage2/.../running.log`
- `__pycache__/`
- `parser/__pycache__/`
- `dataset -> /data/home/longyulei/CoCoSoDa/dataset`

Potentially review before release:

- `parser/my-languages.so` is a compiled binary. It may help reviewers avoid rebuilding tree-sitter, but binary artifacts are often discouraged in source releases. If retained, document platform compatibility. If removed, document how to rebuild it.

## 2. Missing Source, Config, Or Scripts

Required source/scripts for the main experiment appear present.

Missing or incomplete release-support files:

- `requirements.txt` or `environment.yml` is missing.
- No minimal smoke-test script is present.
- No small sample dataset is present. This is acceptable if raw datasets are intentionally excluded, but README should clearly explain where/how to place CodeSearchNet-style JSONL files.
- No explicit license file was found.

## 3. README Coverage

README currently explains:

- Project purpose.
- Core files.
- Expected local runtime assets.
- Main Stage-1, Stage-2, and reranking commands.

README gaps before reviewer release:

- Installation is not sufficiently specified because there is no dependency file.
- Data preparation is described only at the expected-path level; it does not define JSONL fields expected by the code.
- Demo/smoke test is missing.
- The `harvest_retrieval_failures.py` example uses `--loaded_model_filename`, but the script accepts `--checkpoint_file` or `--output_dir`. This README command should be corrected before release.
- Reviewer commands should mention expected GPU/memory assumptions, or provide a CPU/debug alternative.

## 4. Repository-Root Script Execution

The main shell scripts are written to run from the repository root using relative paths:

- `bash run_stage1_experiment.sh`
- `bash run_stage2_experiment.sh`
- `bash run_rerank.sh <lang> <checkpoint>`

They do not require `cd` into the old CoCoSoDa repository.

Release risk:

- The current local `dataset` path is a symlink to `/data/home/longyulei/CoCoSoDa/dataset`. This must not be committed. Reviewers should create their own `dataset/<lang>/...` directory or symlink to their local data.
- Scripts write outputs to `./saved_models/refcode/...`, which is fine for runtime but must remain ignored.

## 5. Hard-Coded Local Paths Or Private Assumptions

No private IPs were found.

Hard-coded absolute/local path currently present:

- `dataset -> /data/home/longyulei/CoCoSoDa/dataset` symlink.

Old-project/local naming assumptions:

- `DeepSoftwareAnalytics/CoCoSoDa` is used as the default Hugging Face model/backbone. This appears intentional for reproduction, but README should state it as an external pretrained model dependency.
- `run_stage1_experiment.sh` defaults `EMBEDDINGS_CACHE` to `dataset/<lang>/train_query_cocosoda_emb.pt`. This is a local cache filename and not required if reviewers rebuild hard negatives. Consider documenting it as optional or renaming for release.

Other external links:

- `run.py` includes public Apache/Hugging Face/apex/WhiteningBERT-related URLs in comments/help text.
- `parser/build.sh` clones public tree-sitter repositories.

## 6. Files That Should Not Be Committed

The following generated files are currently present and should be removed before release:

```text
./__pycache__/model.cpython-39.pyc
./__pycache__/utils.cpython-39.pyc
./parser/__pycache__/DFG.cpython-39.pyc
./parser/__pycache__/__init__.cpython-39.pyc
./parser/__pycache__/utils.cpython-39.pyc
./saved_models/refcode/stage1/ruby_seed123456_lr8e-6_tau0.03_relW0.25_top16_20260703164513/checkpoint-best-mrr/model.bin
./saved_models/refcode/stage1/ruby_seed123456_lr8e-6_tau0.03_relW0.25_top16_20260703164513/running.log
./saved_models/refcode/stage2/ruby_seed123456_K1_W0.45_LI0.05_20260703193709/checkpoint-best-mrr/model.bin
./saved_models/refcode/stage2/ruby_seed123456_K1_W0.45_LI0.05_20260703193709/running.log
```

The following should also not be committed:

- `dataset` symlink.
- Any `*.pkl`, `*.pt`, `*.bin`, `*.npy`, `*.npz`, `*.log`, raw dataset files, wandb outputs, or temporary experiment outputs.

`.gitignore` already excludes the major generated/runtime categories:

- `dataset/`
- `saved_models/`
- `logs/`
- `*.bin`, `*.pt`, `*.pth`, `*.ckpt`, `*.safetensors`
- `*.pkl`, `*.pickle`, `*.npy`, `*.npz`
- `__pycache__/`, `*.pyc`

## 7. Author Identity Or Paper Metadata Exposure

No obvious author name, institution, email, GitHub username, or private server address was found in the tracked source-level files inspected.

Potential metadata/naming notes:

- `DeepSoftwareAnalytics/CoCoSoDa` is a public Hugging Face model identifier, not a private identity leak.
- The local dataset symlink contains the username `longyulei` and old project path. It must not be committed.
- `README.md` uses the project/paper name `ReFCode`; this is expected.

## 8. Requirements Or Environment

No `requirements.txt` or `environment.yml` was found.

Minimum dependencies inferred from imports and current environment:

- Python 3.9+
- `torch`
- `transformers`
- `numpy`
- `tqdm`
- `prettytable`
- `tree_sitter`

Recommended release action:

- Add either `requirements.txt` or `environment.yml`.
- Include CUDA/PyTorch installation guidance separately, because PyTorch wheels vary by CUDA version.

## 9. Minimal Smoke Test

No dedicated smoke-test script currently exists.

Recommended minimal smoke test:

- Create a tiny synthetic CodeSearchNet-style dataset under `tests/fixtures/tiny/<lang>/` or document a reviewer-provided tiny dataset.
- Run `build_global_hard_negatives.py` with `--debug_limit 8`, small `--topk`, and CPU or GPU.
- Run `harvest_retrieval_failures.py` with `--debug_limit 8`, small `--topk`, and a tiny/local checkpoint if available.
- Run `run.py --debug --n_debug_samples 8` for a one-step/short debug pass if the code path supports it.

Because the current scripts are full experiment scripts, reviewers need either a GPU and real data or a smaller smoke command documented in README.

## 10. Exact Reviewer Verification Commands

These commands should be run from the repository root after dependencies are installed and data is prepared.

### Static Checks

```bash
bash -n train_initial_retriever.sh \
  run_stage1_experiment.sh \
  train_failure_calibrated_refinement.sh \
  run_stage2_experiment.sh \
  run_rerank.sh \
  parser/build.sh
```

```bash
python -m py_compile \
  model.py \
  run.py \
  build_global_hard_negatives.py \
  harvest_retrieval_failures.py \
  run_dual_granularity_rerank.py \
  utils.py \
  parser/DFG.py \
  parser/build.py \
  parser/utils.py
```

### Release Hygiene Checks

```bash
find . -type f \( \
  -name "*.bin" -o \
  -name "*.pt" -o \
  -name "*.pth" -o \
  -name "*.ckpt" -o \
  -name "*.safetensors" -o \
  -name "*.pkl" -o \
  -name "*.pickle" -o \
  -name "*.npy" -o \
  -name "*.npz" -o \
  -name "*.log" -o \
  -name "*.pyc" \
\)
```

```bash
find . -maxdepth 3 -type d \( \
  -name "saved_models" -o \
  -name "logs" -o \
  -name "wandb" -o \
  -name "__pycache__" \
\)
```

```bash
find . -maxdepth 2 -type l -printf '%p -> %l\n'
```

```bash
rg -n "(/data/|/home/|longyulei|CoCoSoDa|aaai|AAAI|github|@|http://|https://|[0-9]{1,3}(\.[0-9]{1,3}){3})" \
  . \
  -g '!saved_models/**' \
  -g '!dataset/**' \
  -g '!__pycache__/**'
```

### Main Experiment Reproduction

Prepare the dataset layout:

```bash
mkdir -p dataset/ruby
# Place or symlink:
# dataset/ruby/train.jsonl
# dataset/ruby/valid.jsonl
# dataset/ruby/test.jsonl
# dataset/ruby/codebase.jsonl
```

Run Stage-1 with default Ruby settings:

```bash
bash run_stage1_experiment.sh ruby 123456
```

Run Stage-2 from the latest Stage-1 checkpoint:

```bash
bash run_stage2_experiment.sh ruby 123456
```

Run final dual-granularity reranking:

```bash
bash run_rerank.sh ruby saved_models/refcode/stage2/<run>/checkpoint-best-mrr/model.bin
```

Equivalent explicit Stage-2 failure-harvesting command:

```bash
python harvest_retrieval_failures.py \
  --train_data_file dataset/ruby/train.jsonl \
  --output_file dataset/ruby/self_mined_top32_from_stage1.pkl \
  --model_name_or_path DeepSoftwareAnalytics/CoCoSoDa \
  --config_name DeepSoftwareAnalytics/CoCoSoDa \
  --tokenizer_name DeepSoftwareAnalytics/CoCoSoDa \
  --checkpoint_file saved_models/refcode/stage1/<run>/checkpoint-best-mrr/model.bin
```

## Final Pre-Release To-Do List

- Remove `saved_models/` from the release tree.
- Remove `__pycache__/` and all `*.pyc`.
- Remove the local `dataset` symlink before commit.
- Add `requirements.txt` or `environment.yml`.
- Add or document a minimal smoke test.
- Fix the README Stage-2 failure-harvesting argument from `--loaded_model_filename` to `--checkpoint_file`.
- Decide whether to keep or rebuild `parser/my-languages.so`; document the choice.
- Confirm the final release tree with the hygiene commands above.
