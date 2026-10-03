#!/usr/bin/env python3
"""Audit historical Java/JavaScript backbone-transfer artifacts.

The old artifacts can be provenance-complete while still being ineligible for
the final paper method.  The final method excludes CTRD and LI auxiliary
training, so this auditor reports provenance and method compatibility as two
separate decisions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Any


BACKBONES = ["unixcoder", "codebert", "graphcodebert"]
LANGUAGES = ["java", "javascript"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_train_urls(path: Path) -> list[str]:
    urls = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                urls.append(str(row.get("url", row.get("retrieval_idx", len(urls)))))
    return urls


def audit_candidates(rows: list[Any], urls: list[str]) -> dict[str, int]:
    invalid = self_hits = same_url = duplicates = 0
    min_len = 10**9
    max_len = 0
    for i, raw in enumerate(rows):
        if not isinstance(raw, (list, tuple)):
            invalid += 1
            continue
        min_len = min(min_len, len(raw))
        max_len = max(max_len, len(raw))
        duplicates += len(raw) - len(set(raw))
        for candidate in raw:
            if not isinstance(candidate, int) or candidate < 0 or candidate >= len(urls):
                invalid += 1
                continue
            self_hits += int(candidate == i)
            same_url += int(urls[candidate] == urls[i])
    return {
        "rows": len(rows),
        "min_candidates": 0 if min_len == 10**9 else min_len,
        "max_candidates": max_len,
        "invalid_indices": invalid,
        "self_hits": self_hits,
        "same_url_hits": same_url,
        "duplicate_candidates": duplicates,
    }


def contains_all(text: str, needles: list[str]) -> tuple[bool, list[str]]:
    missing = [needle for needle in needles if needle not in text]
    return not missing, missing


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--legacy-root",
        type=Path,
        default=Path(os.environ.get("COCOSODA_REPO", "../CoCoSoDa")),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("experimental_results/11_backbone_transfer_artifact_audit_seed123456"),
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    legacy = args.legacy_root.resolve()
    output = args.output_root if args.output_root.is_absolute() else repo / args.output_root
    output.mkdir(parents=True, exist_ok=True)

    train_urls = {
        lang: load_train_urls(repo / "dataset" / lang / "train.jsonl")
        for lang in LANGUAGES
    }
    artifacts: dict[str, Any] = {}
    checks: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}

    def check(name: str, passed: bool, detail: Any) -> None:
        checks.append({"name": name, "status": "PASS" if passed else "FAIL", "detail": detail})

    for backbone in BACKBONES:
        artifacts[backbone] = {}
        model_path = legacy / "pretrained_backbones" / f"{backbone}-base-local"
        check(f"{backbone}.local_base_model", (model_path / "config.json").is_file() and (model_path / "pytorch_model.bin").is_file(), str(model_path))
        for lang in LANGUAGES:
            stage1 = legacy / "saved_models" / "backbone_generalization" / f"{backbone}_stage1_ctrd" / f"{lang}_seed123456"
            stage2 = legacy / "saved_models" / "backbone_generalization" / f"{backbone}_refcode_stage2" / f"{lang}_seed123456_K1_W0.45_LI0.05"
            cache = legacy / "dataset" / lang / f"self_mined_top32_from_{backbone}_ctrd.pkl"
            rerank_log = legacy / "logs" / f"{lang}_{backbone}_test_fusion.log"
            original_json = legacy / "saved_models" / "late_interaction_rerank" / f"{lang}_{backbone}_refcode_top50_alpha0.5" / f"late_interaction_{lang}_top50_alpha0.5.json"
            migrated_json = repo / "experimental_results" / "04_late_interaction_rerank_mixed_variants" / f"{lang}_{backbone}_refcode_top50_alpha0.5" / f"late_interaction_{lang}_top50_alpha0.5.json"
            stage1_ckpt = stage1 / "checkpoint-best-mrr" / "model.bin"
            stage2_ckpt = stage2 / "checkpoint-best-mrr" / "model.bin"

            required = [stage1 / "running.log", stage1_ckpt, stage2 / "running.log", stage2_ckpt, cache, rerank_log, original_json, migrated_json]
            check(f"{backbone}.{lang}.required_files", all(path.is_file() for path in required), [str(path) for path in required if not path.is_file()])

            stage1_log = (stage1 / "running.log").read_text(encoding="utf-8", errors="replace")
            stage1_needles = [
                f"train_data_file='dataset/{lang}/train.jsonl'",
                f"eval_data_file='dataset/{lang}/valid.jsonl'",
                f"test_data_file='dataset/{lang}/test.jsonl'",
                f"codebase_file='dataset/{lang}/codebase.jsonl'",
                f"model_name_or_path='{model_path}'",
                "loaded_model_filename=None",
                "num_train_epochs=10",
                "seed=123456",
                "use_ua_bm25_hard_negative=True",
                f"ua_hard_idx_file='dataset/{lang}/aaai_hard_idx_top500_rank50.pkl'",
                "use_ctrd=True",
            ]
            stage1_fields_ok, stage1_missing = contains_all(stage1_log, stage1_needles)
            stage1_order_ok = stage1_log.rfind("Saving model checkpoint") < stage1_log.find("runnning test") and stage1_log.find("runnning test") >= 0
            check(f"{backbone}.{lang}.stage1_provenance", stage1_fields_ok and stage1_order_ok, {"missing": stage1_missing, "last_save_before_test": stage1_order_ok})

            with cache.open("rb") as handle:
                cache_data = pickle.load(handle)
            candidate_audit = audit_candidates(cache_data.get("self_mined_idx", []), train_urls[lang])
            expected_cache_checkpoint = f"./saved_models/backbone_generalization/{backbone}_stage1_ctrd/{lang}_seed123456/checkpoint-best-mrr/model.bin"
            cache_ok = (
                cache_data.get("train_data_file") == f"dataset/{lang}/train.jsonl"
                and cache_data.get("checkpoint") == expected_cache_checkpoint
                and cache_data.get("topk") == 32
                and cache_data.get("exclude_same_url") is True
                and candidate_audit["rows"] == len(train_urls[lang])
                and candidate_audit["min_candidates"] == 32
                and candidate_audit["max_candidates"] == 32
                and all(candidate_audit[key] == 0 for key in ["invalid_indices", "self_hits", "same_url_hits", "duplicate_candidates"])
            )
            check(f"{backbone}.{lang}.candidate_cache", cache_ok, {"metadata": {k: v for k, v in cache_data.items() if k != "self_mined_idx"}, "indices": candidate_audit})

            stage2_log = (stage2 / "running.log").read_text(encoding="utf-8", errors="replace")
            stage2_needles = [
                f"train_data_file='dataset/{lang}/train.jsonl'",
                f"eval_data_file='dataset/{lang}/valid.jsonl'",
                f"test_data_file='dataset/{lang}/test.jsonl'",
                f"codebase_file='dataset/{lang}/codebase.jsonl'",
                f"loaded_model_filename='{expected_cache_checkpoint[:-len('/model.bin')]}/model.bin'",
                f"self_mined_idx_file='dataset/{lang}/self_mined_top32_from_{backbone}_ctrd.pkl'",
                "num_train_epochs=3",
                "seed=123456",
                "use_ctrd=False",
                "use_self_mined_hard_negative=True",
                "self_mined_train_k=1",
                "self_mined_weight=0.45",
                "use_lite_late_interaction_train=1",
                "li_weight=0.05",
            ]
            stage2_fields_ok, stage2_missing = contains_all(stage2_log, stage2_needles)
            stage2_order_ok = stage2_log.rfind("Saving model checkpoint") < stage2_log.find("runnning test") and stage2_log.find("runnning test") >= 0
            check(f"{backbone}.{lang}.stage2_provenance", stage2_fields_ok and stage2_order_ok, {"missing": stage2_missing, "last_save_before_test": stage2_order_ok})

            rerank_text = rerank_log.read_text(encoding="utf-8", errors="replace")
            rerank_needles = [
                f"codebase_file: dataset/{lang}/codebase.jsonl",
                f"eval_data_file: dataset/{lang}/test.jsonl",
                f"loaded_model_filename: ./saved_models/backbone_generalization/{backbone}_refcode_stage2/{lang}_seed123456_K1_W0.45_LI0.05/checkpoint-best-mrr/model.bin",
                f"num_queries: {10955 if lang == 'java' else 3291}",
            ]
            rerank_ok, rerank_missing = contains_all(rerank_text, rerank_needles)
            check(f"{backbone}.{lang}.rerank_provenance", rerank_ok, rerank_missing)

            original_hash = sha256_file(original_json)
            migrated_hash = sha256_file(migrated_json)
            check(f"{backbone}.{lang}.migrated_result_identity", original_hash == migrated_hash, {"original": original_hash, "migrated": migrated_hash})
            hashes[f"{backbone}/{lang}/stage1_checkpoint"] = sha256_file(stage1_ckpt)
            hashes[f"{backbone}/{lang}/candidate_cache"] = sha256_file(cache)
            hashes[f"{backbone}/{lang}/stage2_checkpoint"] = sha256_file(stage2_ckpt)
            hashes[f"{backbone}/{lang}/rerank_json"] = migrated_hash

            result = json.loads(migrated_json.read_text(encoding="utf-8"))
            artifacts[backbone][lang] = {
                "stage1": str(stage1),
                "stage2": str(stage2),
                "candidate_cache": str(cache),
                "candidate_audit": candidate_audit,
                "result": result,
                "provenance_complete": stage1_fields_ok and stage1_order_ok and cache_ok and stage2_fields_ok and stage2_order_ok and rerank_ok and original_hash == migrated_hash,
                "final_method_compatible": False,
                "exclusion_reasons": [
                    "stage-1 source training enables CTRD",
                    "stage-2 refinement enables LI auxiliary training with weight 0.05",
                    "reranking uses legacy K=50 rather than the frozen final K=20 setting",
                ],
            }

    failed = [item for item in checks if item["status"] == "FAIL"]
    provenance_complete = not failed
    manifest = {
        "audit": "historical alternative-backbone transfer artifacts",
        "verdict": "EXCLUDE_EXISTING_ARTIFACTS",
        "provenance_complete": provenance_complete,
        "final_method_compatible": False,
        "reason": "All historical runs use a CTRD source and LI-trained refinement, which conflicts with the frozen final method.",
        "checks_total": len(checks),
        "checks_failed": len(failed),
        "artifacts": artifacts,
        "checks": checks,
        "sha256": dict(sorted(hashes.items())),
        "next_action": "Run one fresh UniXcoder Java+JavaScript transfer with no CTRD, no LI auxiliary training, and frozen K=20/alpha=0.5 inference.",
    }
    (output / "audit_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Historical backbone-transfer artifact audit",
        "",
        "**Decision: exclude all existing transfer scores from the final paper.**",
        "",
        f"Provenance checks: {'complete' if provenance_complete else 'incomplete'} ({len(checks) - len(failed)}/{len(checks)} passed). The exclusion is a method-compatibility decision, not an unidentified-data-provenance failure.",
        "",
        "| Backbone | Languages | Candidate cache | Checkpoint/eval lineage | Final-method compatible | Decision |",
        "|---|---|---|---|---|---|",
    ]
    for backbone in BACKBONES:
        items = artifacts[backbone]
        cache_ok = all(item["candidate_audit"]["invalid_indices"] == 0 and item["candidate_audit"]["same_url_hits"] == 0 for item in items.values())
        lineage_ok = all(item["provenance_complete"] for item in items.values())
        lines.append(f"| {backbone} | Java, JavaScript | {'pass' if cache_ok else 'fail'} | {'pass' if lineage_ok else 'fail'} | No | Exclude |")
    lines += [
        "",
        "Every historical source run enables CTRD, every historical refinement run enables LI auxiliary training (`li_weight=0.05`), and the saved fusion score uses legacy `K=50`. These settings conflict with the frozen no-CTRD/no-LI final method and frozen `K=20, alpha=0.5` inference protocol.",
        "",
        "The six candidate caches themselves are train-only, language-aligned, row-aligned, in range, self-excluding, and same-URL filtered. The migrated JSON files are byte-identical to the original results, and the reranking logs identify the matching test/codebase/checkpoint paths.",
        "",
        "## Next action",
        "",
        "Run one fresh UniXcoder transfer on Java and JavaScript: UniXcoder UA-HN source without CTRD, FC refinement without LI auxiliary training, and the already frozen `K=20, alpha=0.5` parameter-free reranker. Do not spend compute on CodeBERT or GraphCodeBERT unless the UniXcoder result is useful.",
    ]
    (output / "AUDIT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "SHA256SUMS.txt").write_text("\n".join(f"{value}  {key}" for key, value in sorted(hashes.items())) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": manifest["verdict"], "provenance_complete": provenance_complete, "checks_total": len(checks), "checks_failed": len(failed), "output": str(output)}, indent=2))
    return 0 if provenance_complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
