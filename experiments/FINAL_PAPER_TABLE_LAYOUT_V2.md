# NAACL final paper table layout (v2)

This is the authoritative presentation plan.  It replaces the earlier
engineering-oriented table plan.  Each main-paper table answers one review
question; detailed provenance and per-seed outputs belong in the appendix.

## Table 1. Where are the high-ranked failures?

**Question answered:** Is there a concrete, benchmark-grounded ranking problem
to refine?  This is descriptive motivation, not evidence of semantic error.

| Language | # test queries | Gold at rank 1 | Gold at ranks 2--10 | Gold at ranks 11--50 | Gold below 50 |
|---|---:|---:|---:|---:|---:|
| Java | | | | | |
| JavaScript | | | | | |
| Ruby | | | | | |
| Python | | | | | |
| PHP | | | | | |
| Go | | | | | |
| Macro avg. | | | | | |

Use a compact table (or a stacked-bar figure if space is tight).  Do not call
every non-ground-truth candidate a "semantic failure"; call it a *ranking
failure under the benchmark label*.

## Table 2. Unified main results on CodeSearchNet

**Question answered:** Does ReFCode outperform standard and strong code-search
retrievers under one local protocol?  This is the only headline comparison
table.  Every row must use the same preprocessing, candidate corpus, evaluator,
and three matched seeds.  HedgeCode is absent unless fully reproduced.

| Method | Java | JavaScript | Ruby | Python | PHP | Go | Avg. MRR | Avg. R@1 | Avg. R@5 | Avg. R@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CoCoSoDa | | | | | | | | | | |
| CoCoSoDa + UA-HN | | | | | | | | | | |
| CoCoSoDa + UA-HN + CTRD | | | | | | | | | | |
| CoCoSoDa + UA-HN + ReFCode-Global | | | | | | | | | | |
| CoCoSoDa + UA-HN + ReFCode-Fusion | | | | | | | | | | |
| CoCoSoDa + UA-HN + CTRD + ReFCode-Fusion | | | | | | | | | | |

In the paper abbreviate rows after defining them once: `CoCoSoDa`, `+UA-HN`,
`+CTRD`, `+ReFCode-G`, `+ReFCode-F`.  Bold the best *reproduced* result and
underline the best no-CTRD result.  Place `mean +/- sd` in the Avg. MRR column;
put full per-language standard deviations in Appendix Table A3 to keep this
table readable.

The last row is a robustness row, visually separated by a midrule.  It is not
used to define the proposed method.  The three no-CTRD rows are the method's
main narrative: `CoCoSoDa -> UA-HN -> ReFCode`.

## Table 3. What creates the improvement?

**Question answered:** Is the gain caused by failure-calibrated supervision,
rather than an arbitrary extra negative, local interaction, or score fusion?
Use two panels in one table, with one metric block (six-language macro average
MRR, R@1, R@10) so it remains compact.

### Panel A: candidate-source control (global branch only)

Same frozen no-CTRD UA-HN Initial; same optimizer, steps, and **one** explicit
negative per query.  LI is off and fusion is off in every row.

| Refinement supervision | Candidate source | K | MRR | R@1 | R@10 | Delta vs. Initial |
|---|---|---:|---:|---:|---:|---:|
| No refinement | -- | 0 | | | | -- |
| Random | random non-positive | 1 | | | | |
| Static HN | fixed pre-mined negative | 1 | | | | |
| Failure-calibrated (FC) | Initial high-ranked non-positive | 1 | | | | |

### Panel B: selected FC pipeline

Same no-CTRD UA-HN Initial and FC pool.  This panel separates the training
component from the inference-time fusion decision.

| FC objective | Local interaction loss | Inference | MRR | R@1 | R@10 | Delta vs. FC global |
|---|---|---|---:|---:|---:|---:|
| FC | No | global | | | | -- |
| FC | Yes | global | | | | |
| FC | Yes | fusion | | | | |

`Fusion` is not a training ablation.  Its row is last because it evaluates a
deployment choice after the FC and LI training choices are fixed.

## Table 4. Is the result reliable and transferable?

**Question answered:** Is the improvement stable rather than a single-seed,
single-retriever, single-benchmark accident?  Use two panels.

### Panel A: paired-seed significance, main Initial

| Language | UA-HN MRR (mean +/- sd) | ReFCode-Fusion MRR (mean +/- sd) | Delta MRR | 95% paired CI | p-value |
|---|---:|---:|---:|---|---:|
| Java | | | | | |
| JavaScript | | | | | |
| Ruby | | | | | |
| Python | | | | | |
| PHP | | | | | |
| Go | | | | | |
| Macro avg. | | | | | |

Use three paired seeds (123456, 2027, 2028), a query-level paired bootstrap
CI, and a paired permutation/bootstrap test.  This panel is mandatory.

### Panel B: transfer / Initial robustness

Only include a row when training split, candidate corpus, and checkpoint
provenance are audited.  UA-HN+CTRD on CodeSearchNet is always eligible as
Initial robustness; an external test set or UniXcoder row is conditional.

| Setting | Initial | ReFCode variant | Initial MRR | ReFCode MRR | Delta |
|---|---|---|---:|---:|---:|
| CodeSearchNet, stronger Initial | UA-HN+CTRD | Fusion | | | |
| AdvTest (if audited) | UA-HN | Global / Fusion | | | |
| Alternative backbone (if audited) | UniXcoder | Global / Fusion | | | |

If no clean external row exists, rename Panel B to **Robustness to Initial
retriever**, retain only the first row, and state cross-corpus generalization as
a limitation.  Never fill it with reported HedgeCode values.

## Table 5. Effectiveness--efficiency trade-off

**Question answered:** Are gains operationally usable, and what is the cost of
the two stages and the reranking budget?  Alpha is selected once on validation,
then frozen for test.

| Variant | K | alpha | MRR | Mining GPU-h | Refinement GPU-h | Global ms/query | Rerank ms/query | Total ms/query | Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| UA-HN Initial | -- | -- | | -- | -- | | -- | | |
| ReFCode-Global | -- | -- | | | | | -- | | |
| ReFCode-Fusion | 10 | fixed | | | | | | | |
| ReFCode-Fusion | 20 | fixed | | | | | | | |
| ReFCode-Fusion | 50 | fixed | | | | | | | |
| ReFCode-Fusion | 100 | fixed | | | | | | | |

Use a companion figure: x-axis total latency/query, y-axis MRR, one point per
K.  Put alpha and candidate-budget sensitivity curves in the appendix unless
they reveal a key stability result.

## Table 6. What are mined ranking failures?

**Question answered:** Do the FC candidates represent meaningful difficult
alternatives rather than a labeling or mining artifact?  Two annotators label
150--200 candidates sampled with a recorded seed.  A pair of qualitative
examples can appear as a figure/listing, but only as illustrations of this
table.

| Annotation category | Count | Ratio | Median Initial rank | Median rank change after fusion |
|---|---:|---:|---:|---:|
| Correct alternative implementation | | | | |
| Globally relevant but functionally incorrect | | | | |
| Local semantic mismatch | | | | |
| Irrelevant | | | | |
| Uncertain | | | | |
| Cohen's kappa | -- | -- | -- | -- |

## Appendix

| Appendix item | Content |
|---|---|
| A1 | Dataset counts and every preprocessing/candidate/split hash. |
| A2 | FC versus Static HN overlap, score gaps, and rank distributions. |
| A3 | Complete per-language, per-seed Table 2 results. |
| A4 | Validation K/alpha and candidate-budget sensitivity. |
| A5 | Full-test versus failure-subset global/local/fusion behavior. |
| A6 | Candidate, cache, duplicate, and project-overlap audit. |
| A7 | Literature-only comparison: HedgeCode and newer large rerankers, each explicitly marked `reported; not reproduced`. |

## Required experiment order

1. Freeze and audit the unified pipeline.
2. Obtain CoCoSoDa, UA-HN, and main no-CTRD ReFCode rows under the same
   protocol; run paired seeds for UA-HN versus ReFCode-Fusion.
3. Run equal-budget Random/Static/FC controls and FC/LI/Fusion components.
4. Profile K and alpha using validation selection.
5. Audit one external or alternative-backbone result; include it only if clean.
6. Annotate mined candidates if time allows.
