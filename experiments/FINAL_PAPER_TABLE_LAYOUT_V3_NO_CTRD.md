# Final NAACL table layout: backbone-agnostic ReFCode, no CTRD

This is the final presentation and execution decision.  CTRD is removed from
the paper, experiments, and headline method.  ReFCode is presented as a
plug-in *failure-calibrated refinement* method; UA-HN is the source retriever
used for the primary six-language experiment, not a named component of
ReFCode.

## Method and terminology

* **Method:** ReFCode mines high-ranked non-ground-truth candidates from a
  frozen source retriever, then refines it with FC supervision and optional
  local-interaction reranking.
* **Primary source retriever:** CoCoSoDa trained with UA-HN.  State this once
  in Experimental Setup / Implementation Details, not in the title, abstract,
  or method name.
* **Backbone transfer:** A source retriever may instead be pure CoCoSoDa,
  UniXcoder, or CodeBERT.  These are compatibility experiments, not parts of
  the method definition.
* **CTRD:** do not mention it in the paper or tables.
* **HedgeCode:** discuss in Related Work.  State in Experimental Setup that a
  faithful unified rerun was not completed, so its published numbers are not
  placed in any quantitative comparison and no claim against it is made.

Suggested transparent wording: "We do not include HedgeCode in quantitative
comparisons because we could not complete a faithful reproduction under the
unified data and evaluation pipeline; consequently, all numerical comparisons
in this paper are locally reproduced."

## Table 1. High-ranked ranking failures

| Language | # test queries | Gold at rank 1 | Gold at ranks 2--10 | Gold at ranks 11--50 | Gold below 50 |
|---|---:|---:|---:|---:|---:|
| Java | | | | | |
| JavaScript | | | | | |
| Ruby | | | | | |
| Python | | | | | |
| PHP | | | | | |
| Go | | | | | |
| Macro avg. | | | | | |

This is a descriptive research-question table.  It must not interpret
non-ground-truth code as semantically wrong.

## Table 2. Main results on CodeSearchNet

All rows are local reproductions under one protocol.  This is the sole wide
model-comparison table.  `+ ReFCode-G/F` means ReFCode global / selected fusion
on the row's source retriever.

| Method | Java | JavaScript | Ruby | Python | PHP | Go | Avg. MRR | Avg. R@1 | Avg. R@5 | Avg. R@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CoCoSoDa | | | | | | | | | | |
| CoCoSoDa + UA-HN | | | | | | | | | | |
| CoCoSoDa + UA-HN + ReFCode-G | | | | | | | | | | |
| CoCoSoDa + UA-HN + ReFCode-F | | | | | | | | | | |

Rows are conceptually grouped into base retrievers and ReFCode variants by a
midrule.  Define `UA-HN` in the baseline paragraph only.  Report three-seed
mean +/- std in Avg. MRR; detailed per-language deviations go to Appendix A3.

## Table 3. Why does ReFCode improve retrieval?

Two compact panels use macro-average MRR/R@1/R@10.  Both begin with the same
frozen UA-HN source retriever.

### Panel A: candidate-source control

LI and fusion are off.  Every refinement row has one explicit negative/query,
the same optimizer, and the same number of updates.

| Refinement supervision | Candidate source | K | MRR | R@1 | R@10 | Delta vs. source |
|---|---|---:|---:|---:|---:|---:|
| No refinement | -- | 0 | | | | -- |
| Random | random non-positive | 1 | | | | |
| Static HN | fixed pre-mined negative | 1 | | | | |
| ReFCode FC | high-ranked non-positive mined by source retriever | 1 | | | | |

### Panel B: selected ReFCode pipeline

| FC objective | Local interaction | Inference | MRR | R@1 | R@10 | Delta vs. FC global |
|---|---|---|---:|---:|---:|---:|
| FC | No | global | | | | -- |
| FC | Yes | global | | | | |
| FC | Yes | fusion | | | | |

## Table 4. Reliability and backbone generality

### Panel A: paired-seed reliability on the primary source retriever

| Language | Source MRR (mean +/- sd) | ReFCode-F MRR (mean +/- sd) | Delta | 95% paired CI | p-value |
|---|---:|---:|---:|---|---:|
| Java | | | | | |
| JavaScript | | | | | |
| Ruby | | | | | |
| Python | | | | | |
| PHP | | | | | |
| Go | | | | | |
| Macro avg. | | | | | |

Use seeds 123456, 2027, and 2028.  This is mandatory.

### Panel B: source-retriever transfer

| Source retriever | Languages | Seed(s) | Source MRR | + ReFCode-G | + ReFCode-F | Fusion delta |
|---|---|---|---:|---:|---:|---:|
| Pure CoCoSoDa | all 6 | 123456 | | | | |
| UniXcoder | Java + JavaScript | 123456 | | | | |
| CodeBERT (only if audited) | Java + JavaScript | 123456 | | | | |

The pure CoCoSoDa row is P0.  UniXcoder is P1 and only included after a clean
checkpoint/candidate audit.  CodeBERT is optional: do not launch a third new
backbone merely to fill a row.  Clearly state the language subset in the table
instead of presenting it as a six-language result.

## Table 5. Effectiveness--efficiency trade-off

| Variant | K | alpha | MRR | Mining GPU-h | Refinement GPU-h | Global ms/query | Rerank ms/query | Total ms/query | Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Source retriever | -- | -- | | -- | -- | | -- | | |
| ReFCode-G | -- | -- | | | | | -- | | |
| ReFCode-F | 10 | fixed on validation | | | | | | | |
| ReFCode-F | 20 | fixed on validation | | | | | | | |
| ReFCode-F | 50 | fixed on validation | | | | | | | |
| ReFCode-F | 100 | fixed on validation | | | | | | | |

Pair this table with an MRR-versus-total-latency figure.  Put alpha and
candidate-budget sensitivity in the appendix unless a stability result is
particularly important.

## Table 6. What do mined candidates represent?

| Annotation category | Count | Ratio | Median source rank | Median rank change after fusion |
|---|---:|---:|---:|---:|
| Correct alternative implementation | | | | |
| Globally relevant but functionally incorrect | | | | |
| Local semantic mismatch | | | | |
| Irrelevant | | | | |
| Uncertain | | | | |
| Cohen's kappa | -- | -- | -- | -- |

Two annotators label 150--200 sampled candidates.  If time is insufficient,
move this table to the appendix rather than replacing it with two anecdotes.

## Appendix and Related Work

* Appendix A1: data, split, cache, duplicate, and candidate-manifest audit.
* Appendix A2: FC/Static overlap and rank distributions.
* Appendix A3: per-language, per-seed results.
* Appendix A4: K/alpha/candidate-budget sensitivity.
* Appendix A5: full-test and failure-subset behavior.
* Appendix A6: any literature-only comparison.  If HedgeCode is listed, use
  columns `Method | Training setting | Evaluation setting | Status` and place
  `HedgeCode | its published multi-task setting | reported CodeSearchNet setup |
  reported, not reproduced`.  Do not print a numerical ranking beside locally
  reproduced results.

## Time-aware execution order (one H20)

1. **P0 (about 104 GPU-h):** two new UA-HN-to-ReFCode six-language seeds for
   Table 2 and Table 4A.
2. **P0 (about 50--60 GPU-h):** equal-budget Random/Static/FC and FC/LI
   controls for Table 3; reuse verified existing no-CTRD FC artifacts whenever
   their exact configuration matches.
3. **P0 (about 45--55 GPU-h):** one six-language pure-CoCoSoDa-to-ReFCode
   run for Table 4B.
4. **P1 (roughly 10--30 GPU-h after audit):** UniXcoder on Java and
   JavaScript only.  Stop if its provenance or data adaptation is not clean.
5. **P1:** validation-only K/alpha sweep, latency/GPU-hour profiling, paired
   inference statistics, and candidate annotation.

This is roughly 210--250 GPU-h plus evaluation/analysis, leaving schedule
margin.  A fresh six-language UniXcoder or a third backbone is not justified
before the P0 evidence is complete.
