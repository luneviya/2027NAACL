# ReFCode research notes — 2026-09-28

## Active A-D screening — started 2026-09-28

The user authorized validation-only screening on JavaScript and Python in tmux
session `refcode_abcd_jspy`. All runs use seed 123456, the no-CTRD UA-HN
source checkpoint, and no test-set selection. Output root:
`runs/four_scheme_screening_20260928`.

- A: boundary-aware repair plus source-margin preservation; single-vector
  retrieval. JavaScript reuses the completed run by symlink; Python retrains.
- B: A plus selective detached MaxSim-to-global margin distillation on source
  failures and near-boundary examples; inference remains single-vector.
- C: validation-time top-50 reranking that weights query-token evidence by its
  dispersion across retrieved candidates.
- D: validation-time top-50 reranking with four contiguous token-region
  vectors and MaxSim scoring.

C/D tune global fusion alpha only over 0.25/0.5/0.75 on validation. These are
screening implementations, not yet paper claims or final architecture choices.

## Retained candidate and verified evidence

Retain `boundary_preserve` as a small positive JavaScript result. Source:
no-CTRD CoCoSoDa+UA-HN; seed 123456; 3 epochs; learning rate 5e-6;
batch 128; FC pool top-8, one encoded negative per query; LI training off.
All variants independently initialize from the same source checkpoint.
Select checkpoints and variants by validation fusion MRR, top-50, alpha 0.5.

| Variant | Validation global MRR | Validation fusion MRR |
|---|---:|---:|
| 00_baseline | 0.7970871097652703 | 0.7963461454318109 |
| 01_failure_only | 0.7946436034820529 | 0.7940852123885469 |
| 02_boundary | 0.7952714238652061 | 0.7946321482011207 |
| 03_boundary_preserve | 0.7992932922898534 | 0.798296792923067 |

Winner directory: `runs/js_method_search/03_boundary_preserve`.
Summary: `runs/js_method_search/selection_summary.json`.
Winner test MRR: global 0.786; local rerank 0.770; fusion 0.789.
Winner test R@1: global 0.690; local rerank 0.672; fusion 0.693.
Test metrics are rounded to three decimals in the saved result file.
No matched top-8 baseline test was run during this search.

Historical top-1 FC test: global 0.782; fusion 0.788. It is not a matched
top-8 baseline and cannot isolate the new loss's effect.
Historical source global test MRR: 0.777.

Training metadata counts: 58,025 examples; 26,099 benchmark ranking failures;
4,658 correct near-boundary examples; 27,268 stable correct examples.
Failure means an eligible non-gold candidate outranks gold under the source;
it is not a claim that the candidate is semantically invalid.

The pair-selection rule chooses the lowest-scoring still-outranking candidate
within the retained top-8 pool, or the top candidate for a correct query.
Near-boundary threshold 0.02, near-example weight 0.25, pair margin 0.02.
Stable-example source-margin preservation uses tolerance 0.005, weight 0.25;
the whole FC loss is scaled by 0.45.

## Interpretation limits and corrections

- `failure_only` gates the extra FC loss; base contrastive and uncertainty
  objectives still train on all examples. Do not say it trains only failures.
- All four runs disable LI training. Their scores cannot establish whether
  LI training helps. They do measure token reranking without that auxiliary
  loss: the winner's fusion helps test by 0.003, but hurts validation global
  MRR by about 0.000997.
- Preservation is promising, not a proven forgetting mechanism. Measure
  per-query repaired ranks and regressions to test the hypothesis.
- Baseline versus failure-only also changes negative selection and pair
  margin, so the drop cannot be attributed only to failure filtering.
- `02_boundary` versus `03_boundary_preserve` is the cleaner preservation
  comparison: validation fusion delta +0.0036646447219463.
- Earlier SIGIR variants cumulatively added uncertainty, weighted MaxSim,
  routing, and false-negative weighting. Their failure is not an independent
  ablation proving each module useless; source provenance also differs.
- The current evaluator skips reranking when gold is outside top-K. This
  leaves reported rank metrics unchanged in that case, but execution uses
  ground-truth knowledge. Audit this before deployment or latency claims.

## Method and paper diagnosis

Dense retrieval is already a complete inference pipeline: encode the query,
search precomputed code vectors, return ranked code. LI is optional.
Distinguish offline training-negative mining from online retrieval.

Current FC and LI lack a demonstrated shared mechanism: FC chooses training
candidates; LI supplies token evidence at inference. Fixed fusion does not
itself prove LI specifically repairs FC-related errors. MaxSim averages each
query token's best code-token match; it does not explicitly ensure negation,
operator direction, or relationships among constraints are satisfied.

Do not add modules merely to increase apparent complexity. Candidate core
questions: repair versus regression; fine-grained semantic constraints;
transferring reliable token evidence into an efficient global retriever.

## Research options (A-D screening now in progress)

1. Repair with source preservation: retain the successful baseline, consider
   adaptive protection or a small residual adapter. Test repaired/regressed
   query counts and ranking deltas. Inference stays single-vector. Existing
   dense-retrieval forgetting literature limits novelty of generic protection.
2. Selective local-to-global distillation: use a frozen teacher's cached local
   or fused scores on TRAIN candidates only; transfer only label-consistent,
   useful pairwise margins to the global student, alongside preservation.
   Deploy the global student alone. First check whether the teacher repairs
   enough failures; current pure LI is weaker and should not teach blindly.
3. Candidate-difference-aware reranking: compare query evidence against spans
   that distinguish retrieved candidate codes, using a small attention module
   over cached/frozen token features. First verify such functional differences
   occur in actual errors. Avoid treating mere AST parseability as proof of a
   semantics-changing negative. Retains two-stage inference and extra cost.
4. Compact multi-vector representations: a few learned semantic vectors per
   code/query, optional diversity supervision, followed by multi-vector scoring.
   Addresses global pooling's information loss, but needs an honest indexing
   and retrieval design; max-over-vectors cannot simply use a single-vector
   ANN index. Higher implementation cost; semantic roles are not guaranteed.

Priority for discussion: first establish repair/regression behavior; then
evaluate selective local distillation if its teacher has useful corrections.
Candidate-difference modeling is a more substantial alternative when actual
errors show unmet semantic constraints. Do not automatically combine all.
Preserve seed 123456 and validation-only method selection.

## Primary references checked

- ANCE-Tele, EMNLP 2022: https://aclanthology.org/2022.emnlp-main.445/
  Dense-retrieval forgetting and negative-group instability are prior work.
- ColBERT, SIGIR 2020: https://arxiv.org/abs/2004.12832
  Late interaction supports both reranking and specialized direct retrieval.
- TCT-ColBERT, RepL4NLP 2021: https://aclanthology.org/2021.repl4nlp-1.17/
  Distilling MaxSim into dot-product retrieval is prior work.
- Selective Knowledge Distillation, ACL 2026:
  https://aclanthology.org/2026.acl-long.1193/
  Selective distillation also exists in binary code similarity; do not claim
  generic selective distillation as a new invention.
