# Baseline reproduction audit for the NAACL main table

## Decision

The main table should contain prior models that use the same six-language
CodeSearchNet natural-language-to-code retrieval setup. Published values and
local reproductions may appear in the same table, but every published-only row
must carry a `Reported` marker and must not be used for paired significance or
direct local deltas.

## Reproduction status

| Method | Same primary task? | Local assets | Reproduction decision |
|---|---|---|---|
| CodeBERT | Yes | Local backbone; official code available | Report published value initially; local rerun is low priority. |
| GraphCodeBERT | Yes | Local backbone; official code available | Report published value initially; local rerun is low priority. |
| SynCoBERT | Yes | No complete local pipeline | Use published value. |
| UniXcoder | Yes | Local backbone and compatible runner | Published value in the main table; use local runs for backbone-transfer analysis. |
| CodeT5+ | Yes | No cached checkpoint/pipeline for this protocol | Use published value. |
| CodeRetriever | Yes | Public repository does not provide a complete released implementation/checkpoint for the claimed reproduction | Use published value only. |
| Soft-InfoNCE | Yes | No audited local pipeline | Use the value reported in UA-HN. |
| TOSS | Yes | No audited local pipeline | Use the value reported in UA-HN. |
| SEA | Yes | No audited local pipeline | Use the value reported in UA-HN. |
| CoCoSoDa | Yes | Original repository, full data, cached official checkpoint, working environment | Reproduction canceled; use the published value unless explicitly rescheduled. |
| UA-HN | Yes | Existing six-language no-CTRD local runs | Use local results. |
| HedgeCode | Yes | Repository and data are local, but only Ruby derived detection data/checkpoint are complete | Keep the published row unless a full six-language reproduction is completed. |
| CCT-Code | No, not the same primary protocol | Paper only | Exclude: its headline search result is Python AdvTest and its CodeSearchNet analysis is zero-shot transfer. |
| CoRet | No | Paper only | Exclude: repository-level code-editing retrieval on SWE-bench/LCA. |
| CodeRAG-Bench | No | Paper only | Exclude: RAG benchmark rather than paired CodeSearchNet code search. |
| Language-Agnostic Code Embeddings | No, zero-shot analysis | Paper only | Discuss in related work; exclude from the supervised main table. |

## Proposed single main table

Use one table, not separate reported/local panels:

1. CodeBERT (Reported)
2. GraphCodeBERT (Reported)
3. SynCoBERT (Reported)
4. UniXcoder (Reported)
5. CodeT5+ (Reported)
6. CodeRetriever (Reported)
7. Soft-InfoNCE (Reported)
8. TOSS (Reported)
9. SEA (Reported)
10. HedgeCode (Reported, unless fully reproduced)
11. CoCoSoDa (Reported)
12. UA-HN source retriever (Local)
13. ReFCode-G (Local)
14. ReFCode-F (Local)

The main claim is the matched improvement from the local UA-HN source to the
local ReFCode variants. Comparisons with reported rows provide field context,
not paired statistical evidence.
