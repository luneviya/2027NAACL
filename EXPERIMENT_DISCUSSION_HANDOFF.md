# NAACL 2027 ReFCode experiment discussion handoff

Last updated: 2026-09-30 (Asia/Shanghai)

This file is the authoritative starting point for a new conversation devoted
only to experiments.  Some older planning documents still mention multi-seed
runs and LI auxiliary training; those parts are stale and must not override the
decisions below.

## Fixed decisions

- Final method: UA-HN source retriever -> failure-candidate (FC) refinement ->
  optional parameter-free token-level reranking of the global top-K candidates.
- CTRD is removed from the method and final claims.
- LI auxiliary **training** is removed.  Local MaxSim is retained only as an
  inference-time, parameter-free reranking score.
- Fixed seed is 123456.  Do not make a multi-seed stability or significance
  claim, and do not schedule extra seeds merely to populate a table.
- Source-model baseline values for CoCoSoDa, GraphCodeBERT, and UniXcoder are
  taken from their original papers and explicitly marked as reported.
- Backbone generalization trains internal source checkpoints only when needed
  to mine failure candidates, then reports locally evaluated ReFCode-G/F. It
  does not report local source test scores or compute significance against the
  reported source numbers.
- HedgeCode is related-work context only; no reproduction is scheduled.

## Completed experiments

### Main no-CTRD seed-123456 result

- UA-HN source macro MRR: 79.77.
- FC-no-LI global macro MRR: 80.57 (+0.80 points).
- FC-no-LI plus validation-selected parameter-free K=20/alpha=0.5 reranking:
  81.06 (+0.48 over FC global; +1.29 over source).
- The older LI-trained fusion reaches 81.05 and does not outperform the final
  no-LI-training method. It is retained only as removal evidence.

Paper-facing Table 2:

- `experimental_results/08_final_no_li_main_seed123456/TABLE2_FINAL_NO_LI.md`
- `experimental_results/08_final_no_li_main_seed123456/table2_final_no_li.json`

Paper-facing Table 3 copy:

- `experimental_results/06b_table3_no_li_positioning_seed123456/TABLE3_NO_LI_POSITIONING.md`
- `experimental_results/06b_table3_no_li_positioning_seed123456/table3_no_li_positioning.json`

The Table 2 source row now uses the exact per-query source rerun rather than
the three-decimal legacy log. This only changes some displayed per-language
hundredths; source macro MRR remains 79.77.

### Table 3 controls

- Six languages are complete for equal-budget Random, Static HN, and FC (K=1,
  LI training off), plus normal-budget FC-no-LI.
- Equal-budget macro MRR: source 79.77, Random 79.72, Static 79.67, FC 80.12.
- Use only the global/bi-encoder metric for candidate-source controls.
- FC-no-LI checkpoints exist for all six languages under
  `runs/table3_controls/components/fc_no_li/ua_hn/fc/seed_123456/`.

## Completed experiment: corrected Table 5

- tmux session: `refcode_table5_no_li`
- live log: `runs/table5_efficiency_no_li_queue.log`
- raw outputs: `runs/table5_efficiency_no_li/`
- final summaries: `experimental_results/07_table5_efficiency_no_li/`
- checkpoint family: FC-no-LI, seed 123456.
- K grid: 10, 20, 50, 100.
- alpha grid: 0, 0.25, 0.5, 0.75, 1.
- Selection: six-language validation macro MRR; freeze before reporting test.
- Valid/test queries both retrieve against `dataset/<lang>/codebase.jsonl`.
- Every run preflights 100% gold URL coverage and non-empty gold code.
- Latency profiling reranks every query; it does not skip compute based on gold
  membership in top-K.

All six validation languages, all six test languages, and the four exact
JavaScript per-K latency/VRAM profiles are complete. Validation selected
`K=20, alpha=0.5`. The frozen test macro MRR is 81.06, versus 80.57 for the
FC global branch. The final result files are:

- `experimental_results/07_table5_efficiency_no_li/TABLE5_RESULTS.md`
- `experimental_results/07_table5_efficiency_no_li/table5_results.json`
- `experimental_results/07_table5_efficiency_no_li/validation_selection.json`

The stopped invalid pilot is retained only for audit at
`runs/table5_efficiency_invalid_protocol_20260930/`.  Do not report it.  It used
LI-trained checkpoints and incorrectly used the query-only valid JSONL as the
code corpus, producing a degenerate validation MRR of 1.0.

## Work remaining

### Priority 0: reranker-only control on the initial UA-HN retriever

This is the highest-priority missing experiment for the paper.  Apply the same
parameter-free token-level MaxSim reranking procedure directly to the locally
evaluated UA-HN initial retriever.  The purpose is to determine how much of the
final improvement comes from ReFCode training and how much comes from a generic
top-K reranking step.

Report the following 2-by-2 comparison over all six CodeSearchNet languages:

| Training | Inference | Macro MRR | Status |
|---|---|---:|---|
| UA-HN | Global retrieval | 79.77 | Complete |
| UA-HN | Top-K MaxSim fusion | TBD | Missing; run this control |
| ReFCode failure-candidate refinement | Global retrieval | 80.57 | Complete |
| ReFCode failure-candidate refinement | Top-K MaxSim fusion | 81.06 | Complete |

Protocol requirements:

- Use the same data splits, candidate codebases, token scoring implementation,
  and evaluation script as the final ReFCode experiment.
- Select K and fusion weight on the six-language validation macro MRR only,
  using the same grid as the ReFCode sweep; freeze them before test evaluation.
- Rerank every query without checking whether the gold code is in the top-K.
- Preserve the order of candidates outside the reranked top-K and compute
  metrics from the resulting full ranking.
- Report MRR, R@1, R@5, R@10, latency, and peak memory.  Parameter-free does
  not mean cost-free.
- Optionally also report UA-HN with ReFCode's fixed K=20 and alpha=0.5 as a
  same-operator diagnostic, but do not substitute this for validation-based
  selection of the UA-HN reranker.

Paper interpretation rule: ReFCode-Global is the core method and the MaxSim
stage is an optional accuracy--latency trade-off.  Do not claim that the
reranker is specific to ReFCode until this control shows that the refined model
retains a clear advantage over UA-HN under the same reranking budget.

### Completed 1: final no-LI result export

- Table 2 is exported under
  `experimental_results/08_final_no_li_main_seed123456/`.
- The paper-facing Table 3 copy now uses the frozen K=20/alpha=0.5 result.
- The older LI-trained result is preserved only as removal evidence.

### Completed 2: failure-neighborhood and paired-query diagnostic

- Exact source, FC-global, and frozen-fusion ranks are exported for all 52,561
  test queries with 100% gold coverage.
- Source-rank bins are @1, @2--10, @11--50, and >50; source -> FC global ->
  fusion transitions and residual FC top-50 repairs are reported.
- Fusion improves macro MRR by 1.29 points over source with a stratified paired
  query-bootstrap 95% CI of [+1.11, +1.47]. Every language-level fusion CI
  excludes zero.
- These CIs quantify query-sampling uncertainty for fixed seed 123456 only;
  they do not establish training-seed stability.
- Extreme repair/regression examples are exported for audit. Semantic mismatch
  labels are not inferred automatically; use the annotation protocol if such
  category claims are needed.

Result files:

- `experimental_results/09_query_rank_diagnostics_seed123456/QUERY_RANK_DIAGNOSTICS.md`
- `experimental_results/09_query_rank_diagnostics_seed123456/query_rank_diagnostics.json`
- `experimental_results/09_query_rank_diagnostics_seed123456/rank_transitions.jsonl`
- `experimental_results/09_query_rank_diagnostics_seed123456/EXTREME_TRANSITIONS.md`

### Completed 3: split/candidate/cache contamination audit

- The read-only audit passed all 121 checks with zero critical failures; 115
  checks passed and six warnings record exact NL-text duplicates already
  present across the frozen upstream splits.
- Exact URL and repository overlap across train/valid/test is zero in all six
  languages. Exact code-token overlap is also zero. Validation/test gold URL
  coverage in the matching codebase is 100%.
- Source BM25 and FC caches are training-only, language-aligned, row-aligned,
  in range, non-self, and same-URL filtered. Every FC row has 32 unique
  candidates.
- Migrated source checkpoints are byte-identical to their original source-run
  checkpoints. Source and FC logs confirm validation-based checkpoint
  selection; FC training contains no test evaluation.
- Recomputing Table 5 from validation only reproduces `K=20, alpha=0.5`
  exactly. All frozen test grids use matching test queries/codebases and the
  final FC-no-LI checkpoint.

Result files:

- `experimental_results/10_data_candidate_provenance_audit_seed123456/AUDIT_REPORT.md`
- `experimental_results/10_data_candidate_provenance_audit_seed123456/audit_manifest.json`
- `experimental_results/10_data_candidate_provenance_audit_seed123456/SHA256SUMS.txt`
- Reproducer: `experiments/audit_data_candidate_provenance.py`

### In progress 4: alternative-backbone transfer

- The historical UniXcoder/CodeBERT/GraphCodeBERT Java and JavaScript assets
  passed all 39 provenance checks, including training-only candidate caches,
  checkpoint linkage, and matching test/codebase evaluation.
- Nevertheless, all historical transfer scores are excluded: their source
  training enables CTRD, their refinement enables LI auxiliary training with
  weight 0.05, and their saved reranker uses the old K=50 setting. They are not
  compatible with the frozen final method.
- A fresh UniXcoder-only Java/JavaScript transfer queue was launched in tmux
  session `refcode_unixcoder_transfer_no_li`. It runs UA-HN without CTRD,
  FC refinement without LI training, and frozen K=20/alpha=0.5 inference.
- JavaScript runs first, followed by Java. The pipeline is resumable and writes
  `runs/backbone_transfer_no_li/unixcoder/STATUS.txt`; no real-time monitoring
  is scheduled.
- CodeBERT and GraphCodeBERT are not scheduled. Reconsider them only if the
  clean UniXcoder result is useful and a second transfer backbone is necessary.

Audit files:

- `experimental_results/11_backbone_transfer_artifact_audit_seed123456/AUDIT_REPORT.md`
- `experimental_results/11_backbone_transfer_artifact_audit_seed123456/audit_manifest.json`
- `experimental_results/11_backbone_transfer_artifact_audit_seed123456/SHA256SUMS.txt`

Running pipeline:

- Runner: `experiments/run_unixcoder_transfer_no_li.sh`
- Raw outputs: `runs/backbone_transfer_no_li/unixcoder/`
- Planned final export: `experimental_results/12_unixcoder_transfer_no_li_seed123456/`

### Optional 5: mined-candidate annotation

- Double-annotate 150--200 candidates as correct alternative, globally relevant
  but functionally wrong, local semantic mismatch, irrelevant, or uncertain.
- Report counts, ratios, source ranks, rank changes, and Cohen's kappa.
- If time is short, move this to the appendix or limitations; do not replace it
  with two unaudited anecdotes.

## Explicitly not remaining

- No more Table 3 training.
- No CTRD experiment.
- No pure CoCoSoDa reproduction.
- No HedgeCode reproduction.
- No mandatory additional seeds.
- No further method search unless the user explicitly reopens that decision.

## Suggested order

1. When the UniXcoder queue completes, audit its final cache/checkpoint/eval
   lineage and consolidate Table 4B only if the protocol passes.
2. Perform candidate annotation only if submission time permits.

## First message for the new experiment conversation

Use:

`请先完整阅读 EXPERIMENT_DISCUSSION_HANDOFF.md，继续跟踪 Table 5，并只在这个对话讨论实验。`
