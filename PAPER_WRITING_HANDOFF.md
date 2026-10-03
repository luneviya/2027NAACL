# NAACL paper-writing handoff

## Latest update — 2026-09-28 (takes precedence over older plans below)

Current override: the user has returned to the original no-CTRD ReFCode and
will not pursue A-D as paper directions. The verified six-language seed-123456
main results are exported under
`experimental_results/05_refcode_noctrd_main_seed123456`; checkpoint entries
are symlinks to the original runs. Tmux session `refcode_noctrd_table3` is
currently re-exporting audited full metrics and will then resume the unfinished
Table 3 Static, Random, and FC-no-LI runs. Do not launch A-D extensions.

The main-result table is now broadened with same-task published baselines
(CodeBERT, GraphCodeBERT, SynCoBERT, UniXcoder, CodeT5+, CodeRetriever,
Soft-InfoNCE, TOSS, SEA, HedgeCode, and CoCoSoDa), explicitly marked as
reported. The user canceled the six-language CoCoSoDa reproduction; no
CoCoSoDa job is queued. The optional runner is retained at
`experiments/run_reproduce_cocosoda.sh` but must not be launched unless the
user explicitly requests it again. See
`experiments/BASELINE_REPRODUCTION_AUDIT.md` for inclusion decisions. CCT-Code,
CoRet, CodeRAG-Bench, and Language-Agnostic Code Embeddings do not use the same
primary supervised six-language retrieval protocol and are not main-table
baselines.

The user subsequently authorized A-D screening on JavaScript and Python.
The validation-only queue is active in tmux session `refcode_abcd_jspy`;
outputs go to `runs/four_scheme_screening_20260928`. It uses seed 123456,
the no-CTRD UA-HN source, and does not use test data for method selection.
JavaScript A reuses `runs/js_method_search/03_boundary_preserve`; B tests
selective local-to-global distillation with single-vector inference; C tests
candidate-difference-aware top-50 reranking; D tests four-vector top-50
reranking. Python runs the same four screening variants.

The user currently uses seed **123456** and does not want claims requiring
multiple seeds. Older paired-seed requirements below are historical, not
instructions to launch seeds 2027/2028.

The JavaScript search completed on 2026-09-27. Retain
`runs/js_method_search/03_boundary_preserve` as a **promising small-gain
candidate**, not a confirmed cross-language improvement. Its frozen-source
boundary supervision repairs benchmark ranking failures/near-boundary pairs
and penalizes erosion of stable positive margins. All four search variants
disabled LI training and CTRD; validation used fixed top-50, alpha-0.5 fusion.
The winner's validation global/fusion MRR is 0.7992933/0.7982968 versus the
matched baseline's 0.7970871/0.7963461. Only the winner was evaluated on test:
global 0.786, local 0.770, fusion 0.789 (stored to three decimals).

Full evidence, interpretation limits, and candidate research directions:
`experiments/METHOD_RESEARCH_NOTES_20260928.md`.

The user is reconsidering the method's central contribution and the role of
LI, and is open to architecture changes. Table 3 remains paused. Do not
automatically launch the five-language extension.

## Current decision

The paper presents **ReFCode** as a plug-in failure-calibrated refinement
method for code retrieval.  It mines high-ranked non-ground-truth candidates
from a frozen source retriever, refines with FC supervision, optionally adds a
local-interaction training loss, and optionally fuses global and local scores
at inference.

The primary experimental source retriever is CoCoSoDa trained with UA-HN.  Do
not put UA-HN in the title, abstract, or method name.  State it once in
Experimental Setup: "Unless otherwise stated, we instantiate the source
retriever with CoCoSoDa trained using UA-HN."

CTRD is entirely removed from the paper and planned experiments.

## Core story to write

1. **Problem:** a strong code retriever still ranks gold code below a small set
   of high-ranked, non-ground-truth candidates.
2. **Idea:** those model-induced ranking failures give a more targeted training
   signal than random or fixed static negatives.
3. **Method:** mine only from the training split; exclude positives; refine the
   retriever using FC; use local interaction and fusion as separately tested
   optional components.
4. **Claims to establish:** FC beats equal-budget Random/Static HN; gains are
   reliable across paired seeds; source-retriever transfer is possible when a
   clean checkpoint is available; costs are reported explicitly.

Avoid saying a non-ground-truth candidate is "wrong" or a genuine semantic
failure.  Use "benchmark ranking failure" or "high-ranked non-ground-truth
candidate".

## HedgeCode wording

Related Work: describe HedgeCode as a multi-task code-search method related to
CoCoSoDa.  Do not include its score in a quantitative table.

Experimental Setup / Limitations wording:

> We do not include HedgeCode in quantitative comparisons because we could not
> complete a faithful reproduction under the unified data and evaluation
> pipeline. Consequently, all numerical comparisons in this paper are locally
> reproduced, and we make no direct numerical claim against its reported
> results.

If essential, Appendix A6 may include a literature-status row only:
`HedgeCode | reported multi-task setting | reported, not reproduced`.

## Main-paper tables

The final templates are in
`experiments/FINAL_TABLE_LAYOUT_V4_REPORTED_COCOSODA.md`.

* **Table 1:** rank distribution of gold code (@1, @2--10, @11--50, >50), by
  language. Motivation only.
* **Table 2A:** CoCoSoDa original-paper values, labeled `Reported; not rerun`.
  No deltas/significance against it.
* **Table 2B:** locally reproduced UA-HN source, +ReFCode-Global,
  +ReFCode-Fusion across six languages; MRR and mean R@1/R@5/R@10.
* **Table 3A:** equal-budget source control: No refinement, Random, Static HN,
  FC; K=1, LI off, global metric.
* **Table 3B:** FC global without LI, FC global with LI, FC with fusion.
* **Table 4A:** three paired seeds by language: mean+/-std, delta, paired 95%
  CI, p-value.
* **Table 4B:** audited transfer on UniXcoder Java+JavaScript; CodeBERT is
  optional and omitted if provenance is not clean.
* **Table 5:** MRR/cost/latency trade-off over fusion K.
* **Table 6:** double annotation of mined candidates, if time permits;
  otherwise appendix, never replaced by only two anecdotes.

## What already exists

* Migrated source results: `experimental_results/`.
* Verified no-CTRD UA-HN source (seed 123456): six-language avg MRR **79.77**.
* Verified no-CTRD ReFCode global: avg MRR **80.61**.
* Verified no-CTRD ReFCode fusion: avg MRR **81.05**.
* These are single-seed evidence; final main result requires seeds 2027 and
  2028.

## Tonight's execution order

1. `bash experiments/run_paired_seeds.sh --seeds 2027,2028`
2. `bash experiments/run_table3_controls.sh`
3. Run validation-only K/alpha sweep and profile cost/latency.
4. Audit existing UniXcoder/CodeBERT artifacts before launching any transfer.

Relevant scripts have passed Bash syntax checks.  No training was launched by
the planning work.
