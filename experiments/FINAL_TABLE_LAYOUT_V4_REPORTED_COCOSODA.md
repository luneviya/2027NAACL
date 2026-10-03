# Final paper layout: reported CoCoSoDa, reproduced ReFCode

## Non-negotiable comparison rule

CoCoSoDa numbers are reused from its original paper at the user's request.
They are **reported reference values**, not local replications.  Therefore:

1. they are visually separated from local results;
2. no significance test or delta is computed from them;
3. the paper's empirical claim is ReFCode's gain over its reproduced UA-HN
   source retriever, not a new SOTA claim over CoCoSoDa; and
4. the caption names the original paper, its evaluation protocol, and the
   fact that it was not rerun.

CTRD is omitted completely.  No experiment is scheduled to train pure
CoCoSoDa.  The existing generic `run_refinement_from_initial.sh` remains the
entry point for audited UniXcoder/CodeBERT transfer experiments.

## Table 1: High-ranked ranking failures

| Language | # Test queries | Gold @1 | Gold @2--10 | Gold @11--50 | Gold >50 |
|---|---:|---:|---:|---:|---:|
| Java | | | | | |
| JavaScript | | | | | |
| Ruby | | | | | |
| Python | | | | | |
| PHP | | | | | |
| Go | | | | | |
| Macro avg. | | | | | |

## Table 2: CodeSearchNet results

**Panel A. Reported reference (different protocol; not used for claims)**

| Method | Java MRR | JavaScript MRR | Ruby MRR | Python MRR | PHP MRR | Go MRR | Avg. MRR | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| CoCoSoDa (original paper) | | | | | | | | Reported; not rerun |

**Panel B. Unified local reproduction (main evidence)**

| Method | Java MRR | JavaScript MRR | Ruby MRR | Python MRR | PHP MRR | Go MRR | Avg. MRR | Avg. R@1 | Avg. R@5 | Avg. R@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| UA-HN source retriever | 77.20 | 77.72 | 82.16 | 77.17 | 71.91 | 92.43 | 79.77 | 70.72 | 91.28 | 95.10 |
| + ReFCode-Global | 78.14 | 78.36 | 82.84 | 77.95 | 73.10 | 93.03 | 80.57 | 71.90 | 91.52 | 95.17 |
| + ReFCode-Fusion (K=20, alpha=0.5) | 78.57 | 78.95 | 83.51 | 78.30 | 73.58 | 93.41 | 81.06 | 72.64 | 91.68 | 95.11 |

Panel B uses the single fixed seed 123456. Do not report mean +/- std or make a
training-seed stability claim. The text may say that the source retriever is
instantiated with CoCoSoDa+UA-HN, but it must not rename ReFCode as a UA-HN
component.

## Table 3: What creates the improvement?

**Panel A: candidate-source control (global branch, LI off)**

| Supervision | Candidate source | Negatives/query | MRR | R@1 | R@10 | Delta vs. source |
|---|---|---:|---:|---:|---:|---:|
| No refinement | -- | 0 | 79.77 | 70.72 | 95.10 | -- |
| Random | random non-positive | 1 | 79.72 | 70.72 | 95.00 | -0.05 |
| Static HN | fixed pre-mined negative | 1 | 79.67 | 70.65 | 94.88 | -0.10 |
| ReFCode FC | source-retrieved high-ranked non-positive | 1 | 80.12 | 71.55 | 94.58 | +0.35 |

**Panel B: final method and LI-removal evidence (normal FC budget)**

| FC | LI | Inference | MRR | R@1 | R@10 | Delta vs. FC global |
|---|---|---|---:|---:|---:|---:|
| Yes | No | Global | 80.57 | 71.90 | 95.17 | -- |
| Yes | No | Fusion, K=20, alpha=0.5 | 81.06 | 72.64 | 95.11 | +0.48 |
| Yes | Yes | Legacy fusion, K=50, alpha=0.5 (removal audit only) | 81.05 | 72.58 | 95.15 | +0.48 |

## Table 4: Query-level uncertainty and backbone transfer

**Panel A: paired query bootstrap (supporting uncertainty)**

| Language | Source MRR | ReFCode-F MRR | Delta | 95% paired query-bootstrap CI |
|---|---:|---:|---:|---|
| Java | 77.20 | 78.57 | +1.37 | [+1.07, +1.66] |
| JavaScript | 77.72 | 78.95 | +1.23 | [+0.68, +1.78] |
| Ruby | 82.16 | 83.51 | +1.35 | [+0.64, +2.10] |
| Python | 77.17 | 78.30 | +1.13 | [+0.82, +1.44] |
| PHP | 71.91 | 73.58 | +1.68 | [+1.36, +1.99] |
| Go | 92.43 | 93.41 | +0.98 | [+0.76, +1.20] |
| Macro avg. | 79.77 | 81.06 | +1.29 | [+1.11, +1.47] |

This panel quantifies query-sampling uncertainty for the fixed seed 123456; it
does not establish training-seed stability. K=20 and alpha=0.5 were frozen on
validation before this test-query analysis. All six language-level intervals
exclude zero.

**Panel B: transfer (include only audited runs)**

| Source retriever | Languages | Source MRR | ReFCode-G | ReFCode-F | Fusion delta | Status |
|---|---|---:|---:|---:|---:|---|
| UniXcoder | Java + JavaScript | | | | | Local reproduction |
| CodeBERT (optional) | Java + JavaScript | | | | | Local reproduction |

Do not put pure CoCoSoDa here because it is intentionally not run.  Omit any
row whose checkpoint, candidate generation, or evaluation split cannot be
audited.

## Table 5: Effectiveness--efficiency trade-off

| Variant | K | alpha (validation-fixed) | MRR | Mining GPU-h | Refinement GPU-h | Global ms/query | Rerank ms/query | Total ms/query | Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Source retriever | -- | -- | | -- | -- | | -- | | |
| ReFCode-Global | -- | -- | | | | | -- | | |
| ReFCode-Fusion | 10 | | | | | | | | |
| ReFCode-Fusion | 20 | | | | | | | | |
| ReFCode-Fusion | 50 | | | | | | | | |
| ReFCode-Fusion | 100 | | | | | | | | |

## Table 6: Mined-candidate annotation

| Category | Count | Ratio | Median source rank | Median rank change after fusion |
|---|---:|---:|---:|---:|
| Correct alternative implementation | | | | |
| Globally relevant but functionally incorrect | | | | |
| Local semantic mismatch | | | | |
| Irrelevant | | | | |
| Uncertain | | | | |
| Cohen's kappa | -- | -- | -- | -- |

## HedgeCode wording

Related Work: discuss its multi-task code-search formulation and its relation
to CoCoSoDa.  Experimental Setup / Limitations: "We did not include HedgeCode
in quantitative comparisons because a faithful reproduction under our unified
pipeline was not completed.  We therefore make no direct numerical claim
against its reported results."  An appendix may contain a no-score literature
status table with `HedgeCode | reported setting | reported, not reproduced`.
