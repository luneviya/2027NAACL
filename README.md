# ReFCode: Learning from Retriever-Induced Confusions for Code Search

This repository provides the implementation and reproduction scripts for ReFCode.

## Framework

![ReFCode framework](figure/framework.png)

## Environment

Create the environment with Conda:

```bash
conda env create -f environment.yml
conda activate refcode
```

Alternatively, install dependencies with pip:

```bash
pip install -r requirements.txt
```

## Dataset

The experiments use the **CodeSearchNet** benchmark, a widely used natural language code search dataset containing query-code pairs from six programming languages.

The expected dataset layout is:

```text
dataset/
├── ruby/
├── javascript/
├── java/
├── go/
├── php/
└── python/
```

Full datasets are not included in this repository. Place the processed data under the corresponding language directory or configure dataset paths in `configs/*.yaml`.

### CodeSearchNet Statistics

| Language | Train | Valid | Test |
|---|---:|---:|---:|
| Ruby | 24,927 | 1,400 | 1,261 |
| JavaScript | 58,025 | 3,885 | 3,291 |
| Go | 167,288 | 7,325 | 8,122 |
| Python | 251,820 | 13,914 | 14,918 |
| Java | 164,923 | 5,183 | 10,955 |
| PHP | 241,241 | 12,982 | 14,014 |

## Running the Pipeline

Run the following commands from the repository root.

### 1. Initial Retriever

```bash
bash run_initial_retriever.sh --lang javascript
```

This step trains or loads the initial retriever and generates top-ranked retrieval results.

### 2. Failure Harvesting

```bash
bash run_failure_harvesting.sh --lang javascript
```

This step mines failure candidates from the initial retrieval results.

### 3. ReFCode Refinement and Final Reranking

```bash
bash run_refcode.sh --lang javascript
```

This step trains the ReFCode refinement model and performs final reranking/evaluation.

The language argument can be replaced with:

```text
ruby, javascript, java, go, php, python
```

## Configuration

The main configuration files are:

```text
configs/
├── retriever.yaml     # Initial retriever settings
├── harvesting.yaml    # Failure candidate harvesting settings
└── refcode.yaml       # ReFCode refinement and reranking settings
```

Dataset paths, output paths, and training settings can be adjusted in these files.

## Results

### Matched local evaluation on CodeSearchNet (%)

| Method | Java | JavaScript | Ruby | Python | PHP | Go | Macro |
|---|---:|---:|---:|---:|---:|---:|---:|
| UA-HN source | 77.20 | 77.72 | 82.16 | 77.17 | 71.91 | 92.43 | 79.77 |
| ReFCode-Global | 78.14 | 78.36 | 82.84 | 77.95 | 73.10 | 93.03 | 80.57 |
| **ReFCode-Rerank** | **78.57** | **78.95** | **83.51** | **78.30** | **73.58** | **93.41** | **81.06** |

### Average Recall@K (%)

| Method | R@1 | R@5 | R@10 |
|---|---:|---:|---:|
| UA-HN source | 70.72 | 91.28 | 95.10 |
| ReFCode-Global | 71.90 | 91.52 | **95.17** |
| **ReFCode-Rerank** | **72.64** | **91.68** | 95.11 |

The source, global-refinement, and reranking rows use the same frozen data,
candidate codebases, and evaluation implementation. Published reference values
are kept separate in the paper because they were not rerun in this pipeline.

## Paper

The ACL-format bilingual working draft is in
[`paper_workspace/acl_template_bilingual/main.tex`](paper_workspace/acl_template_bilingual/main.tex).
All experiment-section LaTeX is contained directly in that file. The recovered
pre-ACL source and figures are preserved under `paper_workspace/refcode_original/`.

## Experiment scripts and evidence

Additional controlled experiments, audits, exporters, and table-generation
scripts are under `experiments/`. The lightweight result and provenance index is
[`experimental_results/RESULTS_MANIFEST.md`](experimental_results/RESULTS_MANIFEST.md).
Set `PYTHON_BIN`, `COCOSODA_REPO`, and model-path environment variables when the
defaults do not match the local machine.

## Outputs

Generated retrieval results, harvested candidates, checkpoints, reranked outputs, and metric files are saved under `saved_models/` or the output paths specified in `configs/*.yaml`.

The `dataset/` and `saved_models/` directories provide the expected layout only. Large datasets, checkpoints, logs, and generated outputs are ignored by Git.

## Notes

This repository provides the code, configuration files, and placeholder directory structure needed to reproduce the ReFCode pipeline. To reproduce the main results, prepare the processed CodeSearchNet data, configure the paths in `configs/*.yaml`, and run the three pipeline commands in order.
