#!/usr/bin/env python3
"""Select Table 5 settings on validation and summarize frozen test results."""

import argparse
import json
from pathlib import Path


LANGUAGES = ["java", "javascript", "ruby", "python", "php", "go"]
METRICS = ["MRR", "R@1", "R@5", "R@10", "R@50"]


def read(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def mean(values):
    values = list(values)
    return sum(values) / len(values)


def pct(value):
    return f"{100.0 * value:.2f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", default="runs/table5_efficiency_no_li")
    parser.add_argument("--output-root", default="experimental_results/07_table5_efficiency_no_li")
    parser.add_argument("--select-only", action="store_true")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    run_root = repo / args.run_root
    output_root = repo / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    valid = {
        lang: read(run_root / "validation" / lang / "grid_results.json")
        for lang in LANGUAGES
    }
    reference = valid[LANGUAGES[0]]["protocol"]
    ks = [int(k) for k in reference["ks"]]
    alphas = [float(a) for a in reference["alphas"]]
    selection = {"selection_split": "validation", "criterion": "six-language macro MRR", "per_k": {}}
    for k in ks:
        scores = {}
        for alpha in alphas:
            key = f"{alpha:g}"
            scores[key] = mean(valid[lang]["fusion"][str(k)][key]["MRR"] for lang in LANGUAGES)
        best_key = max(scores, key=lambda key: (scores[key], -abs(float(key) - 0.5)))
        selection["per_k"][str(k)] = {
            "alpha": float(best_key),
            "validation_macro_mrr": scores[best_key],
            "all_validation_macro_mrr": scores,
        }
    best_k = max(
        ks,
        key=lambda k: (
            selection["per_k"][str(k)]["validation_macro_mrr"],
            -k,
        ),
    )
    selection["best_configuration"] = {
        "k": best_k,
        "alpha": selection["per_k"][str(best_k)]["alpha"],
        "validation_macro_mrr": selection["per_k"][str(best_k)]["validation_macro_mrr"],
    }
    selection_path = output_root / "validation_selection.json"
    selection_path.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(selection["best_configuration"], indent=2))
    if args.select_only:
        return

    test = {
        lang: read(run_root / "test" / lang / "grid_results.json")
        for lang in LANGUAGES
    }
    profiles = {
        k: read(run_root / "profile" / "javascript" / f"k{k}" / "grid_results.json")
        for k in ks
    }
    global_macro = {
        metric: mean(test[lang]["global"][metric] for lang in LANGUAGES)
        for metric in METRICS
    }
    rows = {}
    for k in ks:
        alpha = selection["per_k"][str(k)]["alpha"]
        alpha_name = f"{alpha:g}"
        per_language = {
            lang: test[lang]["fusion"][str(k)][alpha_name]
            for lang in LANGUAGES
        }
        rows[str(k)] = {
            "alpha": alpha,
            "validation_macro_mrr": selection["per_k"][str(k)]["validation_macro_mrr"],
            "test_macro": {
                metric: mean(per_language[lang][metric] for lang in LANGUAGES)
                for metric in METRICS
            },
            "test_per_language": per_language,
            "javascript_profile": {
                "query_encode_ms_per_query": profiles[k]["timing"]["query_encode_ms_per_query"],
                "global_search_ms_per_query": profiles[k]["timing"]["global_search_ms_per_query"],
                "rerank_ms_per_query": profiles[k]["timing"]["rerank_ms_per_query_at_max_k"],
                "total_ms_per_query": profiles[k]["timing"]["online_total_ms_per_query_at_max_k"],
                "peak_vram_mib": profiles[k]["peak_vram_mib"],
            },
        }

    exported = {
        "protocol": {
            "seed": 123456,
            "model": "no-CTRD ReFCode-FC without LI auxiliary training",
            "selection": "alpha selected independently for each K by six-language validation macro MRR",
            "test": "frozen validation-selected alpha",
            "latency": "JavaScript test set; all queries reranked; batch sizes 128/64; FP16",
        },
        "selection": selection,
        "global_test_macro": global_macro,
        "fusion_rows": rows,
    }
    result_path = output_root / "table5_results.json"
    result_path.write_text(json.dumps(exported, indent=2) + "\n", encoding="utf-8")

    md = [
        "# Table 5: effectiveness--efficiency trade-off",
        "",
        "Effectiveness is the six-language test macro average. Latency and peak VRAM are measured on the JavaScript test set. Alpha is selected using validation data only.",
        "",
        "| Variant | K | alpha | MRR | R@1 | R@5 | R@10 | Query+global ms/q | Rerank ms/q | Total ms/q | Peak VRAM MiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| ReFCode-Global | -- | -- | {pct(global_macro['MRR'])} | {pct(global_macro['R@1'])} | {pct(global_macro['R@5'])} | {pct(global_macro['R@10'])} | -- | -- | -- | -- |",
    ]
    for k in ks:
        row = rows[str(k)]
        metric = row["test_macro"]
        profile = row["javascript_profile"]
        query_global = profile["query_encode_ms_per_query"] + profile["global_search_ms_per_query"]
        md.append(
            f"| ReFCode-Fusion | {k} | {row['alpha']:g} | {pct(metric['MRR'])} | "
            f"{pct(metric['R@1'])} | {pct(metric['R@5'])} | {pct(metric['R@10'])} | "
            f"{query_global:.2f} | {profile['rerank_ms_per_query']:.2f} | "
            f"{profile['total_ms_per_query']:.2f} | {profile['peak_vram_mib']:.0f} |"
        )
    md += [
        "",
        "The profile performs late-interaction computation for every query. It does not use gold-answer membership to skip reranking.",
    ]
    (output_root / "TABLE5_RESULTS.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"saved_to: {result_path}")


if __name__ == "__main__":
    main()
