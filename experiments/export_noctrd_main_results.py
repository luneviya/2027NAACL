"""Export the verified seed-123456 no-CTRD ReFCode main results."""

import argparse
import json
import os
from pathlib import Path


LANGUAGES = ["java", "javascript", "ruby", "python", "php", "go"]


def read_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def only_match(root, pattern):
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one match for {root / pattern}, got {matches}")
    return matches[0]


def mean(values):
    values = list(values)
    return sum(values) / len(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        default="experimental_results/05_refcode_noctrd_main_seed123456",
    )
    parser.add_argument(
        "--legacy-root",
        default=os.environ.get(
            "LEGACY_ROOT",
            "../CoCoSoDa/saved_models/aaai_ua_cocosoda_selfmine_late_lite_ablation",
        ),
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    legacy_root = Path(args.legacy_root).expanduser()
    if not legacy_root.is_absolute():
        legacy_root = (repo / legacy_root).resolve()
    output_root = repo / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    migrated_source = (
        repo / "experimental_results/01_initial_cocosoda_ua_hn_bm25_global_seed123456"
    )
    rerank_root = (
        repo / "experimental_results/04_late_interaction_rerank_mixed_variants"
    )

    exported = {
        "protocol": {
            "seed": 123456,
            "source": "CoCoSoDa + UA-HN, no CTRD",
            "refinement": "ReFCode FC top8/train1 weight0.45 + LI train weight0.05",
            "checkpoint_selection": "validation fusion, top50, alpha0.5",
            "test_rerank": "top50, alpha0.5",
            "ctrd_used": False,
        },
        "languages": {},
    }

    for language in LANGUAGES:
        source_dir = only_match(migrated_source, f"{language}_seed123456_*")
        legacy_dir = legacy_root / f"{language}_woCTRD_K1_W0.45_LI0.05"
        rerank_file = (
            rerank_root
            / f"{language}_abl_woCTRD_top50_alpha0.5"
            / f"late_interaction_{language}_top50_alpha0.5.json"
        )
        checkpoint = legacy_dir / "checkpoint-best-mrr/model.bin"
        for required in [
            source_dir / "result.jsonl",
            legacy_dir / "result.jsonl",
            legacy_dir / "running.log",
            rerank_file,
            checkpoint,
        ]:
            if not required.exists():
                raise FileNotFoundError(required)

        source_result = read_jsonl(source_dir / "result.jsonl")[0]
        global_result = read_jsonl(legacy_dir / "result.jsonl")[0]
        rerank_result = read_json(rerank_file)

        language_dir = output_root / language
        language_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_link = language_dir / "checkpoint-best-mrr"
        legacy_link = language_dir / "legacy_run"
        if not checkpoint_link.exists() and not checkpoint_link.is_symlink():
            checkpoint_link.symlink_to(legacy_dir / "checkpoint-best-mrr")
        if not legacy_link.exists() and not legacy_link.is_symlink():
            legacy_link.symlink_to(legacy_dir)

        exported["languages"][language] = {
            "source": {
                "mrr": float(source_result["eval_mrr"]),
                "r1": float(source_result["R@1"]),
                "r5": float(source_result["R@5"]),
                "r10": float(source_result["R@10"]),
            },
            "refcode_global": {
                "mrr": float(rerank_result["bi_mrr"]),
                "r1": float(global_result["R@1"]),
                "r5": float(global_result["R@5"]),
                "r10": float(global_result["R@10"]),
            },
            "refcode_local": {
                "mrr": float(rerank_result["late_interaction_only_mrr"]),
            },
            "refcode_fusion": {
                "mrr": float(rerank_result["fusion_mrr"]),
                "top_k": int(rerank_result["top_k"]),
                "alpha": float(rerank_result["fusion_alpha"]),
            },
            "provenance": {
                "source_run": str(source_dir),
                "legacy_refcode_run": str(legacy_dir),
                "checkpoint": str(checkpoint),
                "legacy_rerank_json": str(rerank_file),
            },
        }

    for key in ["source", "refcode_global", "refcode_fusion"]:
        exported.setdefault("macro", {})[f"{key}_mrr"] = mean(
            exported["languages"][language][key]["mrr"] for language in LANGUAGES
        )
    for key in ["source", "refcode_global"]:
        for metric in ["r1", "r5", "r10"]:
            exported["macro"][f"{key}_{metric}"] = mean(
                exported["languages"][language][key][metric] for language in LANGUAGES
            )

    json_path = output_root / "main_results.json"
    json_path.write_text(json.dumps(exported, indent=2), encoding="utf-8")

    rows = [
        "# No-CTRD ReFCode main results (seed 123456)",
        "",
        "| Language | UA-HN source | ReFCode-Global | ReFCode-LI | ReFCode-Fusion |",
        "|---|---:|---:|---:|---:|",
    ]
    for language in LANGUAGES:
        item = exported["languages"][language]
        rows.append(
            f"| {language} | {item['source']['mrr']:.4f} | "
            f"{item['refcode_global']['mrr']:.4f} | "
            f"{item['refcode_local']['mrr']:.4f} | "
            f"{item['refcode_fusion']['mrr']:.4f} |"
        )
    rows.append(
        f"| Macro avg. | {exported['macro']['source_mrr']:.4f} | "
        f"{exported['macro']['refcode_global_mrr']:.4f} | -- | "
        f"{exported['macro']['refcode_fusion_mrr']:.4f} |"
    )
    (output_root / "MAIN_RESULTS.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(json.dumps(exported["macro"], indent=2))
    print(f"saved_to: {json_path}")


if __name__ == "__main__":
    main()
