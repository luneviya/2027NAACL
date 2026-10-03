# Current evidence and NAACL paper tables

This plan distinguishes **available artifacts** from **publishable evidence**.
An artifact becomes publishable only after its Initial checkpoint, candidate
pool, test split, and metric invocation are recorded in the same provenance
directory.

| Final paper table | What is already available | What is still needed | Script / output rule |
|---|---|---|---|
| Table 1 — high-ranked failure profile | Seed-123456 FC harvesting/checkpoint lineage is available in the migrated Initial and stage-2 artifacts. | Export counts from split-specific training harvesting; add split/cache audit. | Use FC pools created by `run_refinement_from_initial.sh`; never mine test candidates for training. |
| Table 2 — unified CodeSearchNet main result | Six-language seed-123456 UA-HN+CTRD Initial, stage-2, and candidate fusion JSONs are available. | Two matched seeds; one unified metric invocation per reported row; reproduced external baselines if compared. | `run_paired_seeds.sh --seeds 2027,2028`. Do not mix `result.jsonl` and a different reranking JSON. |
| Table 3A — mechanism: FC versus conventional supervision | Existing static pool and prior static-ablation outputs exist. | Random control and an equal-budget FC/Static comparison. | `run_equal_budget_fc_static.sh` uses K=1 on both arms because the migrated static pool supplies one negative/query. `run_migrated_controls.sh --condition random` adds the Random arm. |
| Table 3B — Initial dependency | UA-HN and UA-HN+CTRD checkpoints are migrated; Java/JavaScript alternative-backbone artifacts exist. | CoCoSoDa-to-ReFCode and UA-HN-to-ReFCode runs under current provenance; audit any reused transfer artifact. | `run_initial_variant.sh --variant cocosoda`, then `run_refinement_from_initial.sh`; for existing UA-HN use `run_migrated_controls.sh --initial ua_hn --condition fc`. |
| Table 4A — reliability | Only seed 123456. | Seeds 2027 and 2028, then query-level bootstrap/permutation on saved predictions. | `run_paired_seeds.sh`; add a result-analysis script only after the prediction export format is frozen. |
| Table 4B — transfer | Java/JavaScript CodeBERT, GraphCodeBERT, and UniXcoder reranking JSONs are migrated. | Verify each checkpoint/candidate provenance; AdvTest evaluation. | Reuse transfer only after audit. Run AdvTest through the same rerank-only command with its own test/codebase paths. |
| Table 5 — effectiveness/efficiency | JavaScript has partial alpha/K outputs. | Validation-selected alpha, K sweep on chosen checkpoint, latency/VRAM/GPU-hour measurements. | `run_rerank_sweep.sh`; profile the selected test run, not a different checkpoint. |
| Table 6 — mined-candidate validity | No labeled evidence. | 150–200 candidates, two annotators, Cohen's kappa. | Manual annotation; store CSV plus sampling seed and candidate IDs. |

## Required runs before October 13

1. `run_equal_budget_fc_static.sh` and Random control from the migrated
   UA-HN+CTRD Initial.
2. FC refinement from the migrated UA-HN Initial.
3. CoCoSoDa Initial plus FC refinement, if the clean base checkpoint is not
   already available.
4. Two paired UA-HN+CTRD Initial-to-ReFCode seeds across all six languages.
5. AdvTest, validation K/alpha selection, efficiency profiling, and annotation.

## Method statement to use in the paper

ReFCode is the stage-2 failure-calibrated refinement procedure: it consumes an
arbitrary Initial retriever, mines high-ranked non-ground-truth training
candidates from that retriever, trains the FC global objective plus the
local-interaction auxiliary objective, and optionally uses top-K fusion at
inference. UA-HN and CTRD are Initial-retriever configurations, not required
ReFCode modules.
