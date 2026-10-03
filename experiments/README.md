# NAACL experiment runners

The runners separate an **initial retriever configuration** from ReFCode's
failure-calibrated refinement. `CTRD` is only an Initial configuration option;
the ReFCode stage trains the self-mined failure loss and the local-interaction
auxiliary loss, then optionally reports top-K fusion reranking.

## Current runs

```bash
# New paired seeds for the main reliability table (do not rerun seed 123456 here).
bash experiments/run_paired_seeds.sh --seeds 2027,2028

# ReFCode from an existing UA-HN checkpoint; one language at a time.
bash experiments/run_refinement_from_initial.sh \
  --lang javascript --seed 123456 --initial-name ua_hn \
  --initial-out /path/to/ua_hn/javascript_run --condition fc

# Static HN control. Pass the precomputed, row-aligned static candidate file.
bash experiments/run_refinement_from_initial.sh \
  --lang javascript --seed 123456 --initial-name ua_hn_ctrd \
  --initial-out /path/to/initial_run --condition static \
  --candidate-file dataset/javascript/static_hard_top32_for_ablation.pkl

# Random-negative control; the runner produces a reproducible candidate pickle.
bash experiments/run_refinement_from_initial.sh \
  --lang javascript --seed 123456 --initial-name ua_hn_ctrd \
  --initial-out /path/to/initial_run --condition random

# K/alpha sweep after choosing the checkpoint; select alpha on validation,
# then rerun only the chosen configuration on test for the paper result.
bash experiments/run_rerank_sweep.sh --lang javascript --checkpoint /path/to/model.bin
```

## Paper-table mapping

| Paper table | Required run/output | Existing evidence | New runner |
|---|---|---|---|
| Table 1: high-ranked failure profile | split-specific failure-harvesting summaries | Existing seed-123456 harvesting artifacts | `run_refinement_from_initial.sh` generates new FC pools |
| Table 2: unified CodeSearchNet results | same protocol initial, global branch, and one selected fusion invocation | Migrated seed-123456 artifacts; do not mix stage-2 and rerank metric files | `run_paired_seeds.sh` |
| Table 3A: FC mechanism | FC vs Static HN vs Random, with LI and fusion fixed | Existing `woFailureGuided_static` artifacts | `run_refinement_from_initial.sh` |
| Table 3B: Initial dependency | CoCoSoDa, UA-HN, UA-HN+CTRD each followed by FC refinement | Existing UA-HN+CTRD; provide each source checkpoint explicitly | `run_initial_variant.sh`, `run_refinement_from_initial.sh` |
| Table 4A: reliability | three paired Initial/ReFCode seeds plus query-level significance | Seed 123456 exists | `run_paired_seeds.sh` for two additional seeds |
| Table 4B: transfer | AdvTest and/or audited alternative backbone | Java/JavaScript transfer artifacts exist | reuse only after protocol audit |
| Table 5: effectiveness/efficiency | top-K/alpha sweep plus profiled latency and memory | Partial rerank JSONs exist | `run_rerank_sweep.sh` |
| Table 6: mined-candidate annotation | double annotation and agreement | none | manual; save annotation CSV outside model outputs |

## Non-negotiable result rule

Each paper row must come from one provenance directory and one evaluation
invocation. Never combine `result.jsonl` from stage-2 with metrics from a
different reranking JSON. Every new runner writes `provenance.txt` next to the
checkpoint and metrics.
