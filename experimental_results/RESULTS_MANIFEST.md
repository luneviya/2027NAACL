# Migrated CoCoSoDa experiment artifacts

All artifacts below were **copied** from the external CoCoSoDa workspace configured by `COCOSODA_REPO`; the source files were not moved or modified. Directory labels are based on each run's `running.log`, not on the former directory names alone.

| Destination directory | Verified configuration | Scope / status |
|---|---|---|
| `01_initial_cocosoda_ua_hn_bm25_global_seed123456` | CoCoSoDa; seed `123456`; 10 epochs; `use_ua_uncertainty=True`; `use_ua_bm25_hard_negative=True`; BM25 hard-index file `aaai_hard_idx_top500_rank50.pkl`; no CTRD flag | Six-language global initial-retriever run. This is the UA-HN-style global baseline. |
| `02_initial_cocosoda_ua_hn_bm25_plus_ctrd_joint_seed123456` | CoCoSoDa; seed `123456`; 10 epochs; the same UA uncertainty and BM25 hard negatives; `use_ctrd=True`, weight `0.25`, batch top-K `16` | Six-language joint UA-HN+CTRD initial retriever. Its log records `loaded_model_filename=None`: it is **not** a continuation checkpoint from directory 01. |
| `03_refcode_self_mined_from_ctrd_lite_late_seed123456` | Loads the matching directory-02 checkpoint; 3 epochs; self-mined candidates from `self_mined_top32_from_ctrd.pkl`; weight `0.45`; retained top-K `8`; lightweight late-interaction training weight `0.05` | Six-language stage-2 result. During this stage, `use_ctrd=False` and BM25 hard negatives are disabled because the model starts from the CTRD checkpoint and refines it with self-mined candidates. |
| `04_late_interaction_rerank_mixed_variants` | JSON outputs record `top_k`, `fusion_alpha`, `bi_mrr`, `late_interaction_only_mrr`, `fusion_mrr`, and query count | This directory is deliberately labelled **mixed variants**, not final results. It contains main six-language reranking candidates, ablations, CodeBERT/GraphCodeBERT/UniXcoder transfer runs, and JavaScript parameter-sensitivity runs. Select paper rows only after matching a JSON subdirectory to its stage-2 checkpoint. |

## Reranking-output selection rule

Do not aggregate JSON files solely by their filenames. The folder includes outputs such as `*_abl_*`, `*_codebert_refcode_*`, `*_graphcodebert_refcode_*`, `*_unixcoder_refcode_*`, and `param_sensitivity/*`, which are not the six-language main result. Treat these as their named ablation, transfer, or sensitivity conditions.

Candidate main-condition subdirectories are named `*_late_lite_*` (for example, `java_late_lite_validfusion_top50_alpha0.5`). Their `bi_mrr` is the stage-2 global branch, while `fusion_mrr` is the top-K, alpha-specific reranking result. This association still needs a final script-level provenance check before publication because the JSON files do not themselves record the source checkpoint path.

## Paper-facing result exports

The entries below supersede older planning documents for paper tables.

| Directory | Role | Status |
|---|---|---|
| `05_refcode_noctrd_main_seed123456` | Older LI-trained no-CTRD result | Audit/removal evidence only; not the final paper method |
| `06_table3_noctrd_seed123456` | Original Table 3 control/component export | Source for the completed equal-budget controls and LI-removal audit |
| `06b_table3_no_li_positioning_seed123456` | Final paper-facing Table 3 | Authoritative; candidate-source controls plus final no-LI method |
| `07_table5_efficiency_no_li` | Corrected effectiveness/efficiency sweep | Authoritative; validation selects `K=20, alpha=0.5` |
| `08_final_no_li_main_seed123456` | Final paper-facing Table 2 | Authoritative; source 79.77, FC global 80.57, frozen fusion 81.06 |
| `09_query_rank_diagnostics_seed123456` | Paired query bootstrap and rank-transition analysis | Authoritative supporting analysis; fusion delta +1.29 points, 95% CI [+1.11, +1.47] |
| `10_data_candidate_provenance_audit_seed123456` | Split/candidate/cache contamination and provenance audit | Authoritative; 121 checks, zero critical failures, six upstream exact-NL-duplicate warnings |
| `11_backbone_transfer_artifact_audit_seed123456` | Historical UniXcoder/CodeBERT/GraphCodeBERT transfer audit | Authoritative exclusion audit; provenance 39/39 passed, but all old scores are incompatible with the no-CTRD/no-LI/K=20 final method |

The final method uses seed 123456, no CTRD, and no LI auxiliary training. Its
parameter-free MaxSim reranker uses `K=20, alpha=0.5`, selected by six-language
validation macro MRR before test reporting. The older LI-trained fusion result
(81.05 macro MRR) is retained only as evidence for removing LI training and
must not be presented as the final method.

The source row in Table 2 now uses the exact per-query source rerun rather than
the three-decimal legacy log. This changes a few displayed per-language
hundredths but leaves the source macro MRR at 79.77. The paired bootstrap and
rank-transition artifacts are under
`experimental_results/09_query_rank_diagnostics_seed123456/`.

The frozen pipeline provenance audit is under
`experimental_results/10_data_candidate_provenance_audit_seed123456/`. Exact
URL, repository, and code-token split overlaps are zero; both hard-negative
cache families are training-only and language-aligned; source checkpoint
copies are SHA-256 identical; and Table 5's validation-only selection was
independently reproduced. The six warnings retain exact NL-text duplicates in
the upstream benchmark as an explicit dataset caveat rather than hiding them.

The historical transfer audit is under
`experimental_results/11_backbone_transfer_artifact_audit_seed123456/`. It
excludes all old transfer scores for method mismatch despite complete
provenance. A fresh UniXcoder-only Java/JavaScript no-CTRD/no-LI run is queued
under `runs/backbone_transfer_no_li/unixcoder/`; it is not an authoritative
paper result until the queue completes and its final lineage audit passes.
