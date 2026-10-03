#!/usr/bin/env python3
"""Audit split isolation, candidate caches, and checkpoint/result provenance.

This is a read-only audit of the frozen seed-123456 ReFCode pipeline.  It does
not train or evaluate a model.  Critical failures make the program exit with a
non-zero status; exact-content duplicates in the upstream dataset are reported
as warnings because they are not created by the experimental pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


LANGUAGES = ["java", "javascript", "ruby", "python", "php", "go"]
SPLITS = ["train", "valid", "test", "codebase"]
EXPECTED_KS = [10, 20, 50, 100]
EXPECTED_ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def token_hash(value: Any) -> bytes | None:
    if value is None:
        return None
    if isinstance(value, list):
        text = "\x1f".join(str(part) for part in value)
    else:
        text = " ".join(str(value).split())
    if not text:
        return None
    return hashlib.sha256(text.encode("utf-8")).digest()


def load_split(path: Path, include_content: bool) -> dict[str, Any]:
    file_digest = hashlib.sha256()
    urls: list[str] = []
    repos: set[str] = set()
    nl_hashes: set[bytes] = set()
    code_hashes: set[bytes] = set()
    partitions: Counter[str] = Counter()
    with path.open("rb") as handle:
        for raw_line in handle:
            file_digest.update(raw_line)
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            row_index = len(urls)
            urls.append(str(row.get("url", row.get("retrieval_idx", row_index))))
            if row.get("repo") is not None:
                repos.add(str(row["repo"]))
            if row.get("partition") is not None:
                partitions[str(row["partition"])] += 1
            if include_content:
                nl = token_hash(row.get("docstring_tokens", row.get("docstring")))
                code = token_hash(row.get("code_tokens", row.get("code")))
                if nl is not None:
                    nl_hashes.add(nl)
                if code is not None:
                    code_hashes.add(code)
    return {
        "count": len(urls),
        "urls": urls,
        "url_set": set(urls),
        "duplicate_url_rows": len(urls) - len(set(urls)),
        "repos": repos,
        "nl_hashes": nl_hashes,
        "code_hashes": code_hashes,
        "partitions": dict(sorted(partitions.items())),
        "sha256": file_digest.hexdigest(),
        "resolved_path": str(path.resolve()),
    }


def compact_split(split: dict[str, Any]) -> dict[str, Any]:
    return {
        key: split[key]
        for key in [
            "count",
            "duplicate_url_rows",
            "partitions",
            "sha256",
            "resolved_path",
        ]
    }


def add_check(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    observed: Any,
    expected: Any,
    severity: str = "critical",
) -> None:
    checks.append(
        {
            "name": name,
            "status": "PASS" if passed else ("WARN" if severity == "warning" else "FAIL"),
            "severity": severity,
            "observed": observed,
            "expected": expected,
        }
    )


def normalize_path(value: str | Path, base: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def one_source_dir(source_root: Path, lang: str) -> Path:
    matches = sorted(source_root.glob(f"{lang}_seed123456_*"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one source run for {lang}, found {len(matches)}")
    return matches[0]


def audit_candidate_indices(
    rows: Iterable[Any], train_urls: list[str], list_valued: bool
) -> dict[str, Any]:
    n = len(train_urls)
    invalid = 0
    self_hits = 0
    same_url_hits = 0
    duplicate_candidates = 0
    lengths: list[int] = []
    row_count = 0
    for i, raw_candidates in enumerate(rows):
        row_count += 1
        candidates = raw_candidates if list_valued else [raw_candidates]
        if not isinstance(candidates, (list, tuple)):
            invalid += 1
            continue
        lengths.append(len(candidates))
        duplicate_candidates += len(candidates) - len(set(candidates))
        for candidate in candidates:
            if not isinstance(candidate, int) or not 0 <= candidate < n:
                invalid += 1
                continue
            if candidate == i:
                self_hits += 1
            if i < n and train_urls[candidate] == train_urls[i]:
                same_url_hits += 1
    return {
        "row_count": row_count,
        "min_candidates_per_row": min(lengths) if lengths else 0,
        "max_candidates_per_row": max(lengths) if lengths else 0,
        "invalid_indices": invalid,
        "self_hits": self_hits,
        "same_url_hits": same_url_hits,
        "duplicate_candidates_within_rows": duplicate_candidates,
    }


def expected_protocol_paths(repo: Path, lang: str, split: str, checkpoint: Path) -> dict[str, Any]:
    return {
        "language": lang,
        "split": split,
        "checkpoint": str(checkpoint.resolve()),
        "query_file": str((repo / "dataset" / lang / f"{split}.jsonl").resolve()),
        "code_file": str((repo / "dataset" / lang / "codebase.jsonl").resolve()),
        "seed": 123456,
    }


def audit_grid(
    repo: Path,
    lang: str,
    split: str,
    checkpoint: Path,
    query_count: int,
    checks: list[dict[str, Any]],
) -> dict[str, Any]:
    path = repo / "runs" / "table5_efficiency_no_li" / ("validation" if split == "valid" else "test") / lang / "grid_results.json"
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    protocol = data["protocol"]
    expected = expected_protocol_paths(repo, lang, split, checkpoint)
    observed = {
        "language": protocol.get("language"),
        "split": protocol.get("split"),
        "checkpoint": str(normalize_path(protocol.get("checkpoint", ""), repo)),
        "query_file": str(normalize_path(protocol.get("query_file", ""), repo)),
        "code_file": str(normalize_path(protocol.get("code_file", ""), repo)),
        "seed": protocol.get("seed"),
    }
    add_check(checks, f"{lang}.{split}.grid_protocol", observed == expected, observed, expected)
    add_check(checks, f"{lang}.{split}.grid_query_count", data.get("num_queries") == query_count, data.get("num_queries"), query_count)
    add_check(checks, f"{lang}.{split}.grid_gold_coverage", data.get("missing_gold") == 0 and protocol.get("gold_url_coverage") == 1.0, {"missing_gold": data.get("missing_gold"), "coverage": protocol.get("gold_url_coverage")}, {"missing_gold": 0, "coverage": 1.0})
    add_check(checks, f"{lang}.{split}.grid_candidates", protocol.get("ks") == EXPECTED_KS and protocol.get("alphas") == EXPECTED_ALPHAS, {"ks": protocol.get("ks"), "alphas": protocol.get("alphas")}, {"ks": EXPECTED_KS, "alphas": EXPECTED_ALPHAS})
    return data


def macro(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values)


def render_report(manifest: dict[str, Any]) -> str:
    verdict = manifest["verdict"]
    failure_count = manifest["summary"]["critical_failures"]
    warning_count = manifest["summary"]["warnings"]
    lines = [
        "# Split / candidate / cache contamination and provenance audit",
        "",
        f"**Verdict: {verdict}.** Critical failures: {failure_count}; warnings: {warning_count}.",
        "",
        "This is a read-only audit of the frozen seed-123456 pipeline. It did not train, tune, or evaluate a model.",
        "",
        "## What was audited",
        "",
        "- Train/validation/test exact URL isolation, repository overlap, and exact normalized NL/code duplicates.",
        "- Validation/test gold URL coverage in each language-specific codebase.",
        "- Source-stage BM25 hard-negative caches and FC self-mined caches: training-only origin, row alignment, in-range indices, self exclusion, and same-URL exclusion.",
        "- Source-to-FC checkpoint lineage, run arguments, validation-only checkpoint selection, and absence of test evaluation in FC training logs.",
        "- Table 5 validation/test protocols and independent recomputation of the validation-selected K and alpha.",
        "- SHA-256 identities for datasets, caches, and source/final checkpoints.",
        "",
        "## Split isolation",
        "",
        "| Language | Train | Valid | Test | Codebase | URL overlap T/V/T | Repo overlap T/V/T | Exact NL overlap T/V/T | Exact code overlap T/V/T | Gold missing V/T |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for lang in LANGUAGES:
        item = manifest["languages"][lang]
        split = item["splits"]
        overlap = item["split_overlap"]
        lines.append(
            f"| {lang} | {split['train']['count']} | {split['valid']['count']} | {split['test']['count']} | {split['codebase']['count']} | "
            f"{overlap['url']['train_valid']}/{overlap['url']['train_test']}/{overlap['url']['valid_test']} | "
            f"{overlap['repo']['train_valid']}/{overlap['repo']['train_test']}/{overlap['repo']['valid_test']} | "
            f"{overlap['nl']['train_valid']}/{overlap['nl']['train_test']}/{overlap['nl']['valid_test']} | "
            f"{overlap['code']['train_valid']}/{overlap['code']['train_test']}/{overlap['code']['valid_test']} | "
            f"{item['gold_coverage']['valid_missing']}/{item['gold_coverage']['test_missing']} |"
        )
    lines += [
        "",
        "T/V/T overlap columns are train-valid / train-test / valid-test. Exact-content overlap is reported as an upstream dataset warning, not silently treated as pipeline-generated contamination.",
        "",
        "## Candidate-cache integrity",
        "",
        "| Language | Source rows | Source invalid/self/same-URL | FC rows | FC candidates per row | FC invalid/self/same-URL/duplicate | Source checkpoint copy |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for lang in LANGUAGES:
        item = manifest["languages"][lang]
        source = item["source_cache"]["index_audit"]
        fc = item["fc_cache"]["index_audit"]
        copied = "identical" if item["checkpoint_lineage"]["source_copy_identical"] else "DIFFERS"
        lines.append(
            f"| {lang} | {source['row_count']} | {source['invalid_indices']}/{source['self_hits']}/{source['same_url_hits']} | "
            f"{fc['row_count']} | {fc['min_candidates_per_row']}-{fc['max_candidates_per_row']} | "
            f"{fc['invalid_indices']}/{fc['self_hits']}/{fc['same_url_hits']}/{fc['duplicate_candidates_within_rows']} | {copied} |"
        )
    selection = manifest["table5_selection"]
    lines += [
        "",
        "## Table 5 selection audit",
        "",
        f"Recomputed from validation only: **K={selection['recomputed']['k']}, alpha={selection['recomputed']['alpha']:g}, macro MRR={selection['recomputed']['validation_macro_mrr']:.12f}**.",
        f"Stored selection: **K={selection['stored']['k']}, alpha={selection['stored']['alpha']:g}, macro MRR={selection['stored']['validation_macro_mrr']:.12f}**.",
        "All frozen test grids were checked only for protocol consistency and gold coverage; test scores did not participate in configuration selection.",
        "",
        "## Provenance conclusion",
        "",
        "The source hard negatives and FC candidates are indexed exclusively into the corresponding language's training file. FC mining records the matching source checkpoint, and the migrated source checkpoint is byte-identical to the original source run. Source and FC checkpoints were selected on validation evaluations; the source test evaluation occurs only after its last checkpoint save, and FC training logs contain no test evaluation. Table 5 then selects K/alpha from the six-language validation macro before frozen test reporting.",
        "",
        "## Caveats",
        "",
        "- Exact NL/code duplicate counts reflect the frozen upstream dataset and are surfaced above. Exact URL and repository split overlap are treated more strictly.",
        "- SHA-256 proves artifact identity and lineage, not semantic correctness of the underlying benchmark annotations.",
        "- This audit validates the existing frozen artifacts; it does not add multi-seed evidence.",
        "",
        "Machine-readable details, every check, and full hashes are in `audit_manifest.json`; hashes are also listed in `SHA256SUMS.txt`.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--original-root",
        type=Path,
        default=Path(os.environ.get("COCOSODA_REPO", "../CoCoSoDa")),
    )
    parser.add_argument("--output-root", type=Path, default=Path("experimental_results/10_data_candidate_provenance_audit_seed123456"))
    args = parser.parse_args()

    repo = args.repo.resolve()
    original_root = args.original_root.resolve()
    output_root = args.output_root if args.output_root.is_absolute() else repo / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    source_root = repo / "experimental_results" / "01_initial_cocosoda_ua_hn_bm25_global_seed123456"
    fc_component_root = repo / "runs" / "table3_controls" / "components" / "fc_no_li"
    fc_root = fc_component_root / "ua_hn" / "fc" / "seed_123456"
    fc_cache_root = fc_component_root / "candidate_pools" / "ua_hn" / "fc" / "seed_123456"

    checks: list[dict[str, Any]] = []
    languages: dict[str, Any] = {}
    artifact_hashes: dict[str, str] = {}
    valid_grids: dict[str, dict[str, Any]] = {}

    for lang in LANGUAGES:
        split_data = {
            split: load_split(repo / "dataset" / lang / f"{split}.jsonl", include_content=split != "codebase")
            for split in SPLITS
        }
        for split, data in split_data.items():
            artifact_hashes[f"dataset/{lang}/{split}.jsonl"] = data["sha256"]

        pairs = [("train", "valid"), ("train", "test"), ("valid", "test")]
        overlap: dict[str, dict[str, int]] = {kind: {} for kind in ["url", "repo", "nl", "code"]}
        for left, right in pairs:
            label = f"{left}_{right}"
            overlap["url"][label] = len(split_data[left]["url_set"] & split_data[right]["url_set"])
            overlap["repo"][label] = len(split_data[left]["repos"] & split_data[right]["repos"])
            overlap["nl"][label] = len(split_data[left]["nl_hashes"] & split_data[right]["nl_hashes"])
            overlap["code"][label] = len(split_data[left]["code_hashes"] & split_data[right]["code_hashes"])
        add_check(checks, f"{lang}.split_url_isolation", all(v == 0 for v in overlap["url"].values()), overlap["url"], "all zero")
        add_check(checks, f"{lang}.split_repo_isolation", all(v == 0 for v in overlap["repo"].values()), overlap["repo"], "all zero")
        duplicate_observed = {"nl": overlap["nl"], "code": overlap["code"]}
        add_check(checks, f"{lang}.exact_content_duplicates", all(v == 0 for kind in duplicate_observed.values() for v in kind.values()), duplicate_observed, "all zero", severity="warning")

        codebase_urls = split_data["codebase"]["url_set"]
        valid_missing = len(split_data["valid"]["url_set"] - codebase_urls)
        test_missing = len(split_data["test"]["url_set"] - codebase_urls)
        add_check(checks, f"{lang}.gold_codebase_coverage", valid_missing == 0 and test_missing == 0, {"valid_missing": valid_missing, "test_missing": test_missing}, {"valid_missing": 0, "test_missing": 0})

        source_dir = one_source_dir(source_root, lang)
        source_ckpt = source_dir / "checkpoint-best-mrr" / "model.bin"
        original_ckpt = original_root / "saved_models" / "aaai_ua_cocosoda_global" / source_dir.name / "checkpoint-best-mrr" / "model.bin"
        fc_dir = fc_root / lang
        final_ckpt = fc_dir / "checkpoint-best-mrr" / "model.bin"

        source_hash = sha256_file(source_ckpt)
        original_hash = sha256_file(original_ckpt)
        final_hash = sha256_file(final_ckpt)
        artifact_hashes[f"source_checkpoint_migrated/{lang}/model.bin"] = source_hash
        artifact_hashes[f"source_checkpoint_original/{lang}/model.bin"] = original_hash
        artifact_hashes[f"fc_checkpoint/{lang}/model.bin"] = final_hash
        add_check(checks, f"{lang}.source_checkpoint_copy", source_hash == original_hash, source_hash, original_hash)

        source_cache_path = original_root / "dataset" / lang / "aaai_hard_idx_top500_rank50.pkl"
        with source_cache_path.open("rb") as handle:
            source_cache = pickle.load(handle)
        source_cache_hash = sha256_file(source_cache_path)
        artifact_hashes[f"source_bm25_cache/{lang}.pkl"] = source_cache_hash
        source_index_audit = audit_candidate_indices(source_cache.get("hard_idx", []), split_data["train"]["urls"], list_valued=False)
        expected_source_meta = {
            "train_data_file": f"dataset/{lang}/train.jsonl",
            "model_name_or_path": "DeepSoftwareAnalytics/CoCoSoDa",
            "topk": 500,
            "bm25_rank": 50,
            "filter_same_url": True,
            "num_examples": split_data["train"]["count"],
        }
        observed_source_meta = {key: source_cache.get(key) for key in expected_source_meta}
        add_check(checks, f"{lang}.source_cache_metadata", observed_source_meta == expected_source_meta, observed_source_meta, expected_source_meta)
        source_indices_ok = (
            source_index_audit["row_count"] == split_data["train"]["count"]
            and source_index_audit["invalid_indices"] == 0
            and source_index_audit["self_hits"] == 0
            and source_index_audit["same_url_hits"] == 0
        )
        add_check(checks, f"{lang}.source_cache_indices", source_indices_ok, source_index_audit, "row-aligned, valid, non-self, different URL")

        fc_cache_path = fc_cache_root / f"{lang}_fc_top32.pkl"
        with fc_cache_path.open("rb") as handle:
            fc_cache = pickle.load(handle)
        fc_cache_hash = sha256_file(fc_cache_path)
        artifact_hashes[f"fc_cache/{lang}.pkl"] = fc_cache_hash
        fc_index_audit = audit_candidate_indices(fc_cache.get("self_mined_idx", []), split_data["train"]["urls"], list_valued=True)
        expected_fc_meta = {
            "train_data_file": f"dataset/{lang}/train.jsonl",
            "topk": 32,
            "exclude_same_url": True,
        }
        observed_fc_meta = {key: fc_cache.get(key) for key in expected_fc_meta}
        cache_checkpoint = normalize_path(fc_cache.get("checkpoint", ""), repo)
        add_check(checks, f"{lang}.fc_cache_metadata", observed_fc_meta == expected_fc_meta and cache_checkpoint == source_ckpt.resolve(), {**observed_fc_meta, "checkpoint": str(cache_checkpoint)}, {**expected_fc_meta, "checkpoint": str(source_ckpt.resolve())})
        fc_indices_ok = (
            fc_index_audit["row_count"] == split_data["train"]["count"]
            and fc_index_audit["min_candidates_per_row"] == 32
            and fc_index_audit["max_candidates_per_row"] == 32
            and fc_index_audit["invalid_indices"] == 0
            and fc_index_audit["self_hits"] == 0
            and fc_index_audit["same_url_hits"] == 0
            and fc_index_audit["duplicate_candidates_within_rows"] == 0
        )
        add_check(checks, f"{lang}.fc_cache_indices", fc_indices_ok, fc_index_audit, "32 unique, valid, non-self, different-URL candidates per training row")

        source_log = (source_dir / "running.log").read_text(encoding="utf-8", errors="replace")
        source_needles = [
            f"train_data_file='dataset/{lang}/train.jsonl'",
            f"eval_data_file='dataset/{lang}/valid.jsonl'",
            f"test_data_file='dataset/{lang}/test.jsonl'",
            f"codebase_file='dataset/{lang}/codebase.jsonl'",
            "loaded_model_filename=None",
            "num_train_epochs=10",
            "seed=123456",
            "use_ua_bm25_hard_negative=True",
            f"ua_hard_idx_file='dataset/{lang}/aaai_hard_idx_top500_rank50.pkl'",
        ]
        source_log_order_ok = (
            "Saving model checkpoint" in source_log
            and "runnning test" in source_log
            and source_log.rfind("Saving model checkpoint") < source_log.find("runnning test")
        )
        source_log_ok = all(needle in source_log for needle in source_needles) and source_log_order_ok
        add_check(checks, f"{lang}.source_run_protocol", source_log_ok, {"missing_fields": [needle for needle in source_needles if needle not in source_log], "last_checkpoint_before_test": source_log_order_ok}, "all required fields; last checkpoint save precedes test")

        provenance = (fc_dir / "provenance.txt").read_text(encoding="utf-8", errors="replace")
        fc_log = (fc_dir / "running.log").read_text(encoding="utf-8", errors="replace")
        relative_source = f"./{source_ckpt.relative_to(repo)}"
        relative_cache = f"./{fc_cache_path.relative_to(repo)}"
        fc_needles = [
            f"train_data_file='dataset/{lang}/train.jsonl'",
            f"eval_data_file='dataset/{lang}/valid.jsonl'",
            f"test_data_file='dataset/{lang}/test.jsonl'",
            f"codebase_file='dataset/{lang}/codebase.jsonl'",
            f"loaded_model_filename='{relative_source}'",
            f"self_mined_idx_file='{relative_cache}'",
            "num_train_epochs=3",
            "seed=123456",
            "use_ctrd=False",
            "use_self_mined_hard_negative=True",
            "self_mined_topk=8",
            "self_mined_train_k=1",
            "self_mined_weight=0.45",
            "use_reliable_local_distillation=0",
            "use_lite_late_interaction_train=0",
            "li_weight=0.0",
            "use_valid_fusion_select=1",
            "valid_fusion_topk=50",
            "valid_fusion_alpha=0.5",
        ]
        fc_log_order_ok = "Running late-fusion evaluation" in fc_log and "Saving model checkpoint" in fc_log and fc_log.find("Running late-fusion evaluation") < fc_log.find("Saving model checkpoint")
        fc_no_test = "Eval test results" not in fc_log and "runnning test" not in fc_log
        fc_log_ok = all(needle in fc_log for needle in fc_needles) and fc_log_order_ok and fc_no_test
        add_check(checks, f"{lang}.fc_run_protocol", fc_log_ok, {"missing_fields": [needle for needle in fc_needles if needle not in fc_log], "validation_before_checkpoint": fc_log_order_ok, "no_test_evaluation": fc_no_test}, "all required fields; validation before checkpoint; no test evaluation")
        provenance_needles = [
            "initial_name=ua_hn",
            f"initial_out=./{source_dir.relative_to(repo)}",
            "condition=fc",
            f"candidate_file={relative_cache}",
            "seed=123456",
            "use_lite_late_interaction_train=0",
            "li_weight=0",
        ]
        add_check(checks, f"{lang}.fc_provenance_file", all(needle in provenance for needle in provenance_needles), [needle for needle in provenance_needles if needle not in provenance], "no missing provenance fields")

        valid_grid = audit_grid(repo, lang, "valid", final_ckpt, split_data["valid"]["count"], checks)
        audit_grid(repo, lang, "test", final_ckpt, split_data["test"]["count"], checks)
        valid_grids[lang] = valid_grid

        languages[lang] = {
            "splits": {split: compact_split(data) for split, data in split_data.items()},
            "split_overlap": overlap,
            "gold_coverage": {"valid_missing": valid_missing, "test_missing": test_missing},
            "source_cache": {
                "path": str(source_cache_path),
                "sha256": source_cache_hash,
                "metadata": observed_source_meta,
                "index_audit": source_index_audit,
            },
            "fc_cache": {
                "path": str(fc_cache_path),
                "sha256": fc_cache_hash,
                "metadata": {**observed_fc_meta, "checkpoint": str(cache_checkpoint)},
                "index_audit": fc_index_audit,
            },
            "checkpoint_lineage": {
                "source_migrated": str(source_ckpt),
                "source_original": str(original_ckpt),
                "source_sha256": source_hash,
                "source_original_sha256": original_hash,
                "source_copy_identical": source_hash == original_hash,
                "fc_checkpoint": str(final_ckpt),
                "fc_checkpoint_sha256": final_hash,
            },
        }

        del split_data
        del source_cache
        del fc_cache

    selection_path = repo / "experimental_results" / "07_table5_efficiency_no_li" / "validation_selection.json"
    with selection_path.open(encoding="utf-8") as handle:
        stored_selection_document = json.load(handle)
    stored = stored_selection_document["best_configuration"]
    all_scores: dict[str, dict[str, float]] = {}
    per_k_best: dict[int, tuple[float, float]] = {}
    for k in EXPECTED_KS:
        scores = {
            f"{alpha:g}": macro(valid_grids[lang]["fusion"][str(k)][f"{alpha:g}"]["MRR"] for lang in LANGUAGES)
            for alpha in EXPECTED_ALPHAS
        }
        best_alpha_key = max(scores, key=lambda key: (scores[key], -abs(float(key) - 0.5)))
        all_scores[str(k)] = scores
        per_k_best[k] = (float(best_alpha_key), scores[best_alpha_key])
    best_k = max(EXPECTED_KS, key=lambda k: (per_k_best[k][1], -k))
    recomputed = {
        "k": best_k,
        "alpha": per_k_best[best_k][0],
        "validation_macro_mrr": per_k_best[best_k][1],
    }
    selection_equal = (
        stored.get("k") == recomputed["k"]
        and float(stored.get("alpha")) == recomputed["alpha"]
        and abs(float(stored.get("validation_macro_mrr")) - recomputed["validation_macro_mrr"]) < 1e-15
        and stored_selection_document.get("selection_split") == "validation"
    )
    add_check(checks, "table5.validation_selection_recomputation", selection_equal, recomputed, stored)

    critical_failures = sum(check["status"] == "FAIL" for check in checks)
    warnings = sum(check["status"] == "WARN" for check in checks)
    verdict = "PASS" if critical_failures == 0 else "FAIL"
    manifest = {
        "audit": "seed-123456 split/candidate/cache contamination and provenance",
        "verdict": verdict,
        "summary": {
            "critical_failures": critical_failures,
            "warnings": warnings,
            "checks_total": len(checks),
            "checks_passed": sum(check["status"] == "PASS" for check in checks),
        },
        "scope": {
            "repo": str(repo),
            "original_source_workspace": str(original_root),
            "seed": 123456,
            "languages": LANGUAGES,
            "read_only": True,
        },
        "languages": languages,
        "table5_selection": {
            "selection_split": stored_selection_document.get("selection_split"),
            "criterion": stored_selection_document.get("criterion"),
            "stored": stored,
            "recomputed": recomputed,
            "all_validation_macro_mrr": all_scores,
        },
        "checks": checks,
        "artifact_hashes": dict(sorted(artifact_hashes.items())),
    }
    manifest_path = output_root / "audit_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output_root / "AUDIT_REPORT.md").write_text(render_report(manifest), encoding="utf-8")
    hash_lines = [f"{digest}  {name}" for name, digest in sorted(artifact_hashes.items())]
    (output_root / "SHA256SUMS.txt").write_text("\n".join(hash_lines) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": verdict, **manifest["summary"], "output_root": str(output_root)}, indent=2))
    return 0 if critical_failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
