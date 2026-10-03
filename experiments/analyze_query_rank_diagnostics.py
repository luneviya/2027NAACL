#!/usr/bin/env python3
"""Analyze paired source/global/fusion ranks for the fixed seed-123456 run."""

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LANGUAGES = ["java", "javascript", "ruby", "python", "php", "go"]
DISPLAY_NAMES = {
    "java": "Java",
    "javascript": "JavaScript",
    "ruby": "Ruby",
    "python": "Python",
    "php": "PHP",
    "go": "Go",
}
SOURCE_RANK_BINS = ["1", "2-10", "11-50", ">50"]
K = 20
ALPHA_KEY = "0.5"
BOOTSTRAP_SEED = 123456


def read_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def reciprocal_ranks(ranks):
    values = np.asarray(ranks, dtype=np.float64)
    if np.any(values <= 0):
        raise ValueError("all ranks must be positive")
    return 1.0 / values


def metrics(ranks):
    values = np.asarray(ranks, dtype=np.int64)
    rr = reciprocal_ranks(values)
    return {
        "MRR": float(rr.mean()),
        "R@1": float((values <= 1).mean()),
        "R@5": float((values <= 5).mean()),
        "R@10": float((values <= 10).mean()),
        "R@50": float((values <= 50).mean()),
    }


def paired_bootstrap(source_rr, global_rr, fusion_rr, rng, repetitions):
    deltas = np.stack([global_rr - source_rr, fusion_rr - source_rr])
    n_queries = source_rr.shape[0]
    draws = np.empty((2, repetitions), dtype=np.float64)
    chunk_size = max(1, min(256, 1_500_000 // n_queries))
    for begin in range(0, repetitions, chunk_size):
        end = min(begin + chunk_size, repetitions)
        indices = rng.integers(
            0, n_queries, size=(end - begin, n_queries), dtype=np.int32
        )
        draws[:, begin:end] = deltas[:, indices].mean(axis=2)
    return {"global": draws[0], "fusion": draws[1]}


def interval(draws):
    low, high = np.percentile(draws, [2.5, 97.5])
    return [float(low), float(high)]


def transition_counts(source, target):
    source = np.asarray(source)
    target = np.asarray(target)
    source_failures = source > 1
    return {
        "improved": int((target < source).sum()),
        "tied": int((target == source).sum()),
        "regressed": int((target > source).sum()),
        "source_failures": int(source_failures.sum()),
        "improved_among_source_failures": int(
            ((target < source) & source_failures).sum()
        ),
        "repaired_to_rank1": int(((source > 1) & (target == 1)).sum()),
        "lost_from_rank1": int(((source == 1) & (target > 1)).sum()),
    }


def source_rank_bin(rank):
    if rank == 1:
        return "1"
    if rank <= 10:
        return "2-10"
    if rank <= 50:
        return "11-50"
    return ">50"


def query_text_by_url(language):
    path = ROOT / "dataset" / language / "test.jsonl"
    texts = {}
    for row in read_jsonl(path):
        text = row.get("docstring") or " ".join(row.get("docstring_tokens", []))
        texts[row["url"]] = " ".join(text.split())
    return texts


def pct(value):
    return f"{100.0 * value:.2f}"


def signed_pct(value):
    return f"{100.0 * value:+.2f}"


def ci_text(values):
    return f"[{100.0 * values[0]:+.2f}, {100.0 * values[1]:+.2f}]"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-root", default="runs/query_rank_diagnostics_seed123456"
    )
    parser.add_argument(
        "--output-root",
        default="experimental_results/09_query_rank_diagnostics_seed123456",
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    args = parser.parse_args()

    run_root = (ROOT / args.run_root).resolve()
    output_root = (ROOT / args.output_root).resolve()
    frozen_table2 = read_json(
        ROOT
        / "experimental_results/08_final_no_li_main_seed123456/table2_final_no_li.json"
    )

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    language_results = {}
    bootstrap_draws = {}
    all_transitions = []
    checkpoint_paths = {"source": {}, "final": {}}

    for language in LANGUAGES:
        source_dir = run_root / "source" / language
        final_dir = run_root / "final" / language
        source_grid = read_json(source_dir / "grid_results.json")
        final_grid = read_json(final_dir / "grid_results.json")
        source_rows = read_jsonl(source_dir / "query_ranks.jsonl")
        final_rows = read_jsonl(final_dir / "query_ranks.jsonl")

        if source_grid["protocol"]["global_only"] is not True:
            raise RuntimeError(f"source run is not global-only: {language}")
        if final_grid["protocol"]["global_only"] is not False:
            raise RuntimeError(f"final run unexpectedly global-only: {language}")
        for grid in [source_grid, final_grid]:
            protocol = grid["protocol"]
            if protocol["seed"] != 123456 or protocol["split"] != "test":
                raise RuntimeError(f"unexpected protocol for {language}: {protocol}")
            if grid["missing_gold"] != 0 or protocol["gold_url_coverage"] != 1.0:
                raise RuntimeError(f"gold coverage failed for {language}")
        if len(source_rows) != len(final_rows) or len(source_rows) != source_grid["num_queries"]:
            raise RuntimeError(f"query count mismatch for {language}")

        source_urls = [row["url"] for row in source_rows]
        final_urls = [row["url"] for row in final_rows]
        if source_urls != final_urls or len(source_urls) != len(set(source_urls)):
            raise RuntimeError(f"query URL order/uniqueness mismatch for {language}")

        source_ranks = np.asarray(
            [row["global_rank"] for row in source_rows], dtype=np.int64
        )
        global_ranks = np.asarray(
            [row["global_rank"] for row in final_rows], dtype=np.int64
        )
        fusion_ranks = np.asarray(
            [row["fusion_rank"][str(K)][ALPHA_KEY] for row in final_rows],
            dtype=np.int64,
        )
        source_metrics = metrics(source_ranks)
        global_metrics = metrics(global_ranks)
        fusion_metrics = metrics(fusion_ranks)

        table2_rows = frozen_table2["table2_rows"]
        frozen_source = table2_rows["UA-HN source"]["per_language"][language]
        frozen_global = table2_rows["ReFCode-FC global"]["per_language"][language]
        frozen_fusion = table2_rows["ReFCode-FC + frozen reranker"]["per_language"][language]
        if abs(source_metrics["MRR"] - frozen_source["MRR"]) > 0.00051:
            raise RuntimeError(
                f"source MRR does not reproduce rounded record for {language}: "
                f"{source_metrics['MRR']} vs {frozen_source['MRR']}"
            )
        for name, reproduced, frozen in [
            ("global", global_metrics, frozen_global),
            ("fusion", fusion_metrics, frozen_fusion),
        ]:
            for metric in ["MRR", "R@1", "R@5", "R@10", "R@50"]:
                if abs(reproduced[metric] - frozen[metric]) > 1e-12:
                    raise RuntimeError(
                        f"{name} {metric} mismatch for {language}: "
                        f"{reproduced[metric]} vs {frozen[metric]}"
                    )

        source_rr = reciprocal_ranks(source_ranks)
        global_rr = reciprocal_ranks(global_ranks)
        fusion_rr = reciprocal_ranks(fusion_ranks)
        draws = paired_bootstrap(
            source_rr,
            global_rr,
            fusion_rr,
            rng,
            args.bootstrap_repetitions,
        )
        bootstrap_draws[language] = draws

        bins = {}
        for label in SOURCE_RANK_BINS:
            mask = np.asarray([source_rank_bin(int(rank)) == label for rank in source_ranks])
            bins[label] = {
                "count": int(mask.sum()),
                "source_mrr": float(source_rr[mask].mean()),
                "global_mrr": float(global_rr[mask].mean()),
                "fusion_mrr": float(fusion_rr[mask].mean()),
                "global_delta_mrr": float((global_rr[mask] - source_rr[mask]).mean()),
                "fusion_delta_mrr": float((fusion_rr[mask] - source_rr[mask]).mean()),
            }

        residual_top50 = (global_ranks >= 2) & (global_ranks <= 50)
        language_results[language] = {
            "num_queries": len(source_rows),
            "source": source_metrics,
            "global": global_metrics,
            "fusion": fusion_metrics,
            "global_delta_mrr": float((global_rr - source_rr).mean()),
            "fusion_delta_mrr": float((fusion_rr - source_rr).mean()),
            "global_delta_ci95": interval(draws["global"]),
            "fusion_delta_ci95": interval(draws["fusion"]),
            "source_to_global_transitions": transition_counts(source_ranks, global_ranks),
            "source_to_fusion_transitions": transition_counts(source_ranks, fusion_ranks),
            "global_to_fusion_transitions": transition_counts(global_ranks, fusion_ranks),
            "global_residual_top50": {
                "count": int(residual_top50.sum()),
                "improved_by_fusion": int(
                    ((fusion_ranks < global_ranks) & residual_top50).sum()
                ),
                "repaired_to_rank1": int(
                    ((fusion_ranks == 1) & residual_top50).sum()
                ),
                "unchanged": int(
                    ((fusion_ranks == global_ranks) & residual_top50).sum()
                ),
                "regressed": int(
                    ((fusion_ranks > global_ranks) & residual_top50).sum()
                ),
            },
            "source_rank_bins": bins,
        }
        checkpoint_paths["source"][language] = source_grid["protocol"]["checkpoint"]
        checkpoint_paths["final"][language] = final_grid["protocol"]["checkpoint"]

        for index, url in enumerate(source_urls):
            all_transitions.append(
                {
                    "language": language,
                    "query_index": index,
                    "url": url,
                    "source_rank": int(source_ranks[index]),
                    "global_rank": int(global_ranks[index]),
                    "fusion_rank": int(fusion_ranks[index]),
                    "source_rank_bin": source_rank_bin(int(source_ranks[index])),
                    "global_rr_delta_vs_source": float(global_rr[index] - source_rr[index]),
                    "fusion_rr_delta_vs_source": float(fusion_rr[index] - source_rr[index]),
                }
            )

    macro = {}
    for system in ["source", "global", "fusion"]:
        macro[system] = {
            metric: float(
                np.mean(
                    [language_results[lang][system][metric] for lang in LANGUAGES]
                )
            )
            for metric in ["MRR", "R@1", "R@5", "R@10", "R@50"]
        }
    macro["global_delta_mrr"] = macro["global"]["MRR"] - macro["source"]["MRR"]
    macro["fusion_delta_mrr"] = macro["fusion"]["MRR"] - macro["source"]["MRR"]
    for target in ["global", "fusion"]:
        macro_draws = np.stack(
            [bootstrap_draws[lang][target] for lang in LANGUAGES]
        ).mean(axis=0)
        macro[f"{target}_delta_ci95"] = interval(macro_draws)

    pooled_bins = {}
    for label in SOURCE_RANK_BINS:
        rows = [row for row in all_transitions if row["source_rank_bin"] == label]
        source_ranks = np.asarray([row["source_rank"] for row in rows])
        global_ranks = np.asarray([row["global_rank"] for row in rows])
        fusion_ranks = np.asarray([row["fusion_rank"] for row in rows])
        pooled_bins[label] = {
            "count": len(rows),
            "source_mrr": metrics(source_ranks)["MRR"],
            "global_mrr": metrics(global_ranks)["MRR"],
            "fusion_mrr": metrics(fusion_ranks)["MRR"],
            "global_delta_mrr": float(
                (reciprocal_ranks(global_ranks) - reciprocal_ranks(source_ranks)).mean()
            ),
            "fusion_delta_mrr": float(
                (reciprocal_ranks(fusion_ranks) - reciprocal_ranks(source_ranks)).mean()
            ),
        }

    output_root.mkdir(parents=True, exist_ok=True)
    transition_path = output_root / "rank_transitions.jsonl"
    with transition_path.open("w", encoding="utf-8") as handle:
        for row in all_transitions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    exported = {
        "protocol": {
            "seed": 123456,
            "split": "test",
            "comparison": "paired by query URL",
            "fusion_k": K,
            "fusion_alpha": float(ALPHA_KEY),
            "selection": "frozen on validation before this test analysis",
            "bootstrap": (
                f"paired nonparametric query bootstrap; {args.bootstrap_repetitions} "
                f"repetitions; RNG seed {BOOTSTRAP_SEED}"
            ),
            "macro_bootstrap": "stratified by language, then equal-weight macro average",
            "scope_limit": "query-sampling uncertainty only; not training-seed stability",
        },
        "per_language": language_results,
        "macro": macro,
        "pooled_source_rank_bins": pooled_bins,
        "provenance": {
            "run_root": str(run_root),
            "source_checkpoints": checkpoint_paths["source"],
            "final_checkpoints": checkpoint_paths["final"],
            "per_query_transitions": str(transition_path),
        },
    }
    json_path = output_root / "query_rank_diagnostics.json"
    json_path.write_text(json.dumps(exported, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Query-level uncertainty and rank-transition diagnostics",
        "",
        "All comparisons use test-query pairs under seed 123456. Fusion uses the "
        "validation-frozen K=20 and alpha=0.5 setting; no test-time tuning was performed.",
        "",
        "## Paired query bootstrap",
        "",
        "| Language | Source MRR | FC global MRR | Global delta [95% CI] | Fusion MRR | Fusion delta [95% CI] |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for language in LANGUAGES:
        row = language_results[language]
        lines.append(
            f"| {DISPLAY_NAMES[language]} | {pct(row['source']['MRR'])} | "
            f"{pct(row['global']['MRR'])} | {signed_pct(row['global_delta_mrr'])} "
            f"{ci_text(row['global_delta_ci95'])} | {pct(row['fusion']['MRR'])} | "
            f"{signed_pct(row['fusion_delta_mrr'])} {ci_text(row['fusion_delta_ci95'])} |"
        )
    lines.append(
        f"| Macro avg. | {pct(macro['source']['MRR'])} | {pct(macro['global']['MRR'])} | "
        f"{signed_pct(macro['global_delta_mrr'])} {ci_text(macro['global_delta_ci95'])} | "
        f"{pct(macro['fusion']['MRR'])} | {signed_pct(macro['fusion_delta_mrr'])} "
        f"{ci_text(macro['fusion_delta_ci95'])} |"
    )
    lines.extend(
        [
            "",
            "CIs quantify test-query sampling uncertainty at the fixed training seed; they "
            "do not establish training-seed stability.",
            "",
            "## Source-to-fusion rank transitions",
            "",
            "| Language | Queries | Source failures (rank > 1) | Improved | Tied | Regressed | Repaired to rank 1 | Lost from rank 1 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for language in LANGUAGES:
        row = language_results[language]
        counts = row["source_to_fusion_transitions"]
        lines.append(
            f"| {DISPLAY_NAMES[language]} | {row['num_queries']} | {counts['source_failures']} | "
            f"{counts['improved']} | {counts['tied']} | {counts['regressed']} | "
            f"{counts['repaired_to_rank1']} | {counts['lost_from_rank1']} |"
        )
    lines.extend(
        [
            "",
            "## FC residual top-50 errors and fusion repair",
            "",
            "| Language | FC ranks 2-50 | Improved by fusion | Repaired to rank 1 | Unchanged | Regressed |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for language in LANGUAGES:
        row = language_results[language]["global_residual_top50"]
        lines.append(
            f"| {DISPLAY_NAMES[language]} | {row['count']} | "
            f"{row['improved_by_fusion']} | {row['repaired_to_rank1']} | "
            f"{row['unchanged']} | {row['regressed']} |"
        )
    lines.extend(
        [
            "",
            "Only candidates inside the validation-frozen top-20 neighborhood can be "
            "reordered; ranks 21-50 are included in the residual denominator but remain unchanged.",
            "",
            "## Failure neighborhood by source rank (pooled diagnostic)",
            "",
            "| Source-rank stratum | Queries | Source MRR | FC global MRR | Global delta | Fusion MRR | Fusion delta |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for label in SOURCE_RANK_BINS:
        row = pooled_bins[label]
        lines.append(
            f"| {label} | {row['count']} | {pct(row['source_mrr'])} | "
            f"{pct(row['global_mrr'])} | {signed_pct(row['global_delta_mrr'])} | "
            f"{pct(row['fusion_mrr'])} | {signed_pct(row['fusion_delta_mrr'])} |"
        )
    lines.extend(
        [
            "",
            "The source-rank table is a pooled diagnostic rather than a language-balanced "
            "headline metric. Full per-language strata and all query transitions are in the JSON artifacts.",
            "",
        ]
    )
    (output_root / "QUERY_RANK_DIAGNOSTICS.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    examples = [
        "# Extreme rank transitions (diagnostic examples)",
        "",
        "Examples are selected after evaluation only to inspect the largest reciprocal-rank "
        "changes; they are not used for model or hyperparameter selection.",
        "",
    ]
    for language in LANGUAGES:
        texts = query_text_by_url(language)
        rows = [row for row in all_transitions if row["language"] == language]
        best = sorted(rows, key=lambda row: row["fusion_rr_delta_vs_source"], reverse=True)[:3]
        worst = sorted(rows, key=lambda row: row["fusion_rr_delta_vs_source"])[:3]
        examples.extend([f"## {DISPLAY_NAMES[language]}", "", "| Direction | Query | Source -> Global -> Fusion | RR delta |", "|---|---|---:|---:|"])
        for direction, selected in [("repair", best), ("regression", worst)]:
            for row in selected:
                text = texts[row["url"]][:180].replace("|", "\\|")
                examples.append(
                    f"| {direction} | {text} | {row['source_rank']} -> "
                    f"{row['global_rank']} -> {row['fusion_rank']} | "
                    f"{row['fusion_rr_delta_vs_source']:+.4f} |"
                )
        examples.append("")
    (output_root / "EXTREME_TRANSITIONS.md").write_text(
        "\n".join(examples), encoding="utf-8"
    )

    print(
        json.dumps(
            {
                "saved_to": str(json_path),
                "macro_source_mrr": macro["source"]["MRR"],
                "macro_global_mrr": macro["global"]["MRR"],
                "macro_fusion_mrr": macro["fusion"]["MRR"],
                "macro_global_delta_ci95": macro["global_delta_ci95"],
                "macro_fusion_delta_ci95": macro["fusion_delta_ci95"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
