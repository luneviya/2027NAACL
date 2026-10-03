# Final NAACL experiment and table plan

## Fixed method boundary

The **main pipeline** is deliberately simple and reproducible:

`UA-HN Initial (without CTRD) -> ReFCode FC+LI refinement -> optional top-K fusion`

UA-HN+CTRD is an Initial-retriever **robustness condition only**.  It is not a
component of ReFCode and must not be included in the headline result.  This
keeps the incremental gain attributable to failure-calibrated refinement rather
than to a stronger jointly-trained Initial model.

HedgeCode is neither rerun nor included in a main table.  If needed for
positioning, cite its published result in related work or an appendix as
`reported (not reproduced under our protocol)`; do not calculate a SOTA margin
against it.

## Run order

| Priority | Run | Scope | Purpose | Script |
|---|---|---|---|---|
| P0 | Audit the existing no-CTRD seed | 6 languages, seed 123456 | Freeze the already available UA-HN / FC / fusion rows and their provenance. | `experimental_results/` manifest + metric audit |
| P0 | Matched seeds | seeds 2027, 2028 x 6 languages | Establish reliability for Initial and FC refinement with identical seeds. | `run_paired_seeds.sh --seeds 2027,2028` |
| P0 | Mechanism controls | seed 123456 x 6 languages | Compare FC with Static and Random under the same UA-HN Initial and candidate budget K=1. | `run_equal_budget_fc_static.sh`; `run_migrated_controls.sh --condition random` |
| P1 | Fusion selection and cost | validation plus all 6 test languages | Select K/alpha on validation only; measure GPU-hours, latency, and VRAM. | `run_rerank_sweep.sh` + profiling |
| P1 | CTRD robustness audit | seed 123456 x 6 languages | Show the conclusion persists with a stronger Initial; no new run if provenance is sound. | migrated results |
| P2 | External generalization | one clean external split or audited alternative backbone | Address generalization only when split/candidate provenance is verified. | existing transfer artifacts or a clean new evaluation |
| P2 | Candidate annotation | 150--200 samples | Substantiate that mined candidates are semantically challenging. | double annotation + kappa |

Do **not** run a fresh CoCoSoDa-only branch merely to fill a table: it changes
the central question and consumes time better spent on paired seeds and
controls.  Do **not** launch HedgeCode reproduction for this submission.

## Final paper tables

### Table 1 — Training-candidate protocol and failure profile

Purpose: make the source of supervision auditable before showing gains.

| Language | Train queries | Initial source | Mining split | Candidate rank band | K | Positives excluded? | Test used in mining? |
|---|---:|---|---|---|---:|---|---|
| Java | ... | UA-HN | train only | top-32 non-ground-truth | 8 (main) / 1 (control) | Yes | No |
| ... | | | | | | | |

Add a compact panel/figure with the rank histogram of gold answers before
refinement and the overlap between FC candidates and Static HN.  This replaces
unsupported qualitative claims with a directly checkable protocol.

### Table 2 — Main CodeSearchNet result (the headline table)

Rows are *only* locally reproduced, no-CTRD results.  Report mean and standard
deviation across three paired seeds; the first seed is 123456.

| Method | Java MRR | Java R@1 | Java R@5 | JavaScript MRR | Python MRR | PHP MRR | Go MRR | Ruby MRR | 6-lang avg. MRR | \u0394 vs UA-HN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| UA-HN Initial | ... | ... | ... | ... | ... | ... | ... | ... | ... | -- |
| + ReFCode global (FC) | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| + ReFCode fusion (selected on validation) | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

Footnote: identical seeds, data preprocessing, candidate universe, and
evaluation invocation across all three rows.  For seed 123456, the available
no-CTRD fusion average is **81.05** MRR, versus **79.77** for UA-HN Initial;
the final table must replace these single-seed figures with mean +/- sd.

### Table 3 — Is the gain specifically due to failure-calibrated supervision?

All rows start from the *same frozen UA-HN seed-123456 Initial*, use LI and the
same training schedule, and use exactly one negative/query.  This is the
causal mechanism table.

| Supervision for refinement | Candidate source | Negatives/query | Java MRR | JavaScript MRR | Python MRR | PHP MRR | Go MRR | Ruby MRR | Avg. MRR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| No refinement | -- | 0 | ... | ... | ... | ... | ... | ... | ... |
| Random | random non-positive | 1 | ... | ... | ... | ... | ... | ... | ... |
| Static HN | pre-mined static | 1 | ... | ... | ... | ... | ... | ... | ... |
| ReFCode FC | Initial top-ranked non-positive | 1 | ... | ... | ... | ... | ... | ... | ... |

Use global-branch MRR here, not fusion: otherwise an inference-time score
mixture confounds the training-supervision comparison.

### Table 4 — Robustness to the Initial retriever (not a main ablation)

This table neutralizes the concern that the result is specific to a very strong
or CTRD-enhanced Initial.  It reports seed 123456 unless additional seeds are
available and labels that fact.

| Initial configuration | CTRD? | Initial avg. MRR | + FC global avg. MRR | + fusion avg. MRR | Global gain | Fusion gain |
|---|---|---:|---:|---:|---:|---:|
| UA-HN (main) | No | 79.77 | 80.61 | 81.05 | +0.84 | +1.28 |
| UA-HN + CTRD | Yes | ... | ... | 81.14 | ... | ... |

Interpretation: CTRD changes the starting point, while FC gains remain.  Do not
describe CTRD as part of the proposed method.

### Table 5 — Reliability and efficiency

The statistical unit is the test query, with paired predictions from the same
seed.  Report 95% bootstrap CI and paired permutation/bootstrap p-value for
FC global versus UA-HN, and fusion versus UA-HN.

| System | Seeds | Avg. MRR mean +/- sd | 95% paired CI of delta | p-value | Initial train GPU-h | Mining GPU-h | Refinement GPU-h | Test latency/query | Peak VRAM |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| UA-HN Initial | 3 | ... | -- | -- | ... | -- | -- | ... | ... |
| + ReFCode global | 3 | ... | ... | ... | -- | ... | ... | ... | ... |
| + ReFCode fusion | 3 | ... | ... | ... | -- | -- | -- | ... | ... |

### Table 6 — Generalization check (only if provenance passes audit)

Do not include this table unless the external data split, candidate corpus, and
checkpoint are all verifiable.  One clean result is better than a mixed table.

| Evaluation setting | Initial | + ReFCode | Delta | Candidate corpus | Training/test separation |
|---|---:|---:|---:|---|---|
| Audited alternative backbone or external test set | ... | ... | ... | ... | verified |

If it cannot be completed cleanly, replace the table with a limitation:
"we evaluate CodeSearchNet six-language retrieval and leave cross-corpus
generalization to future work."  Never fill this table with reported HedgeCode
numbers.

### Appendix tables

* A1: Detailed per-language, per-seed results behind Tables 2 and 5.
* A2: Fusion K/alpha validation sweep; clearly identify the chosen setting.
* A3: Candidate annotation protocol, label distribution, and Cohen's kappa.
* A4: Literature-only comparison. HedgeCode and recent large rerankers may
  appear here, each marked `reported` and with dataset/evaluation differences.

## Claims supported by this plan

1. ReFCode improves a strong, reproducible UA-HN Initial on all six
   CodeSearchNet languages under one protocol.
2. The improvement is attributable to failure-calibrated candidates rather
   than merely adding negatives or using static hard negatives.
3. The result is reliable across seeds and has a stated computational cost.
4. The method is compatible with a stronger CTRD-enhanced Initial, but does
   not depend on CTRD.
