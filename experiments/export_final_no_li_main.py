#!/usr/bin/env python3
"""Export the frozen seed-123456 no-LI Table 2 result.

The source row comes from the audited per-query source rerun. The FC-global
and fusion rows come from the corrected Table 5 evaluation, whose K/alpha pair
was selected on six-language validation macro MRR.
"""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LANGUAGES = ["java", "javascript", "ruby", "python", "php", "go"]
METRICS = ["MRR", "R@1", "R@5", "R@10", "R@50"]
OUT = ROOT / "experimental_results/08_final_no_li_main_seed123456"


def read(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def mean(values):
    values = list(values)
    return sum(values) / len(values)


def macro(per_language, metrics):
    return {
        metric: mean(per_language[lang][metric] for lang in LANGUAGES)
        for metric in metrics
    }


def pct(value):
    return f"{100.0 * value:.2f}"


def main():
    table3_path = (
        ROOT
        / "experimental_results/06_table3_noctrd_seed123456/table3_results.json"
    )
    table5_path = (
        ROOT / "experimental_results/07_table5_efficiency_no_li/table5_results.json"
    )
    table3 = read(table3_path)
    table5 = read(table5_path)

    selected = table5["selection"]["best_configuration"]
    selected_k = int(selected["k"])
    selected_alpha = float(selected["alpha"])
    alpha_key = f"{selected_alpha:g}"
    if selected_k != 20 or selected_alpha != 0.5:
        raise RuntimeError(f"Unexpected frozen setting: {selected}")

    source_raw = table3["panel_a_candidate_source"]["Source (no refinement)"]
    source = {}
    source_result_paths = {}
    for lang in LANGUAGES:
        source_path = (
            ROOT
            / "runs/query_rank_diagnostics_seed123456/source"
            / lang
            / "grid_results.json"
        )
        source_result = read(source_path)
        protocol = source_result["protocol"]
        if (
            protocol["seed"] != 123456
            or protocol["split"] != "test"
            or protocol["global_only"] is not True
            or source_result["missing_gold"] != 0
        ):
            raise RuntimeError(f"Unexpected source protocol in {source_path}")
        source[lang] = {
            metric: float(source_result["global"][metric]) for metric in METRICS
        }
        legacy_mrr = float(source_raw["per_language"][lang]["MRR"])
        if abs(source[lang]["MRR"] - legacy_mrr) > 0.00051:
            raise RuntimeError(
                f"Exact source rerun does not match rounded legacy MRR for {lang}"
            )
        source_result_paths[lang] = str(source_path)
    global_rows = {}
    fusion_rows = {}
    raw_result_paths = {}
    checkpoint_paths = {}
    for lang in LANGUAGES:
        result_path = (
            ROOT / "runs/table5_efficiency_no_li/test" / lang / "grid_results.json"
        )
        result = read(result_path)
        protocol = result["protocol"]
        if protocol["seed"] != 123456 or protocol["split"] != "test":
            raise RuntimeError(f"Unexpected protocol in {result_path}: {protocol}")
        if protocol["gold_url_coverage"] != 1.0 or protocol["empty_gold_code"] != 0:
            raise RuntimeError(f"Failed corpus preflight in {result_path}")
        if result["missing_gold"] != 0:
            raise RuntimeError(f"Missing gold entries in {result_path}")
        if Path(protocol["code_file"]).name != "codebase.jsonl":
            raise RuntimeError(f"Unexpected code corpus in {result_path}")

        global_rows[lang] = {
            metric: float(result["global"][metric]) for metric in METRICS
        }
        fusion_rows[lang] = {
            metric: float(result["fusion"][str(selected_k)][alpha_key][metric])
            for metric in METRICS
        }
        raw_result_paths[lang] = str(result_path)
        checkpoint_paths[lang] = protocol["checkpoint"]

    rows = {
        "UA-HN source": {
            "training": "source retriever",
            "inference": "global retrieval",
            "per_language": source,
            "macro": macro(source, METRICS),
        },
        "ReFCode-FC global": {
            "training": "failure-candidate refinement; LI auxiliary disabled",
            "inference": "global retrieval",
            "per_language": global_rows,
            "macro": {
                metric: float(table5["global_test_macro"][metric])
                for metric in METRICS
            },
        },
        "ReFCode-FC + frozen reranker": {
            "training": "failure-candidate refinement; LI auxiliary disabled",
            "inference": (
                f"parameter-free token MaxSim reranking; K={selected_k}; "
                f"alpha={selected_alpha:g}"
            ),
            "per_language": fusion_rows,
            "macro": {
                metric: float(
                    table5["fusion_rows"][str(selected_k)]["test_macro"][metric]
                )
                for metric in METRICS
            },
        },
    }
    source_mrr = rows["UA-HN source"]["macro"]["MRR"]
    for row in rows.values():
        row["delta_mrr_vs_source"] = row["macro"]["MRR"] - source_mrr

    legacy_li = table3["panel_b_components"]["FC, LI on, fusion"]
    final_mrr = rows["ReFCode-FC + frozen reranker"]["macro"]["MRR"]
    legacy_li_mrr = float(legacy_li["macro"]["MRR"])
    exported = {
        "protocol": {
            "seed": 123456,
            "source": "CoCoSoDa + UA-HN, no CTRD",
            "refinement": "failure-candidate refinement without LI auxiliary training",
            "selection": "K/alpha frozen by six-language validation macro MRR",
            "selected_k": selected_k,
            "selected_alpha": selected_alpha,
            "validation_macro_mrr": float(selected["validation_macro_mrr"]),
            "statistical_scope": "single fixed seed; no multi-seed stability claim",
        },
        "table2_rows": rows,
        "discarded_li_training_audit": {
            "status": "removal evidence only; not the paper method",
            "legacy_li_trained_fusion_macro_mrr": legacy_li_mrr,
            "final_no_li_fusion_macro_mrr": final_mrr,
            "li_minus_no_li_mrr": legacy_li_mrr - final_mrr,
        },
        "provenance": {
            "source_control_export": str(table3_path),
            "exact_source_grid_results": source_result_paths,
            "table5_summary": str(table5_path),
            "validation_selection": str(
                ROOT
                / "experimental_results/07_table5_efficiency_no_li/validation_selection.json"
            ),
            "test_grid_results": raw_result_paths,
            "fc_no_li_checkpoints": checkpoint_paths,
        },
    }

    OUT.mkdir(parents=True, exist_ok=True)
    json_path = OUT / "table2_final_no_li.json"
    json_path.write_text(json.dumps(exported, indent=2) + "\n", encoding="utf-8")

    md = [
        "# Table 2: final no-LI ReFCode result (seed 123456)",
        "",
        (
            f"The parameter-free reranker uses K={selected_k} and "
            f"alpha={selected_alpha:g}, selected by six-language validation macro MRR "
            f"({pct(float(selected['validation_macro_mrr']))})."
        ),
        "",
        "All values are percentages. Results use one fixed seed; no multi-seed stability claim is made.",
        "",
        "| Method | Java MRR | JavaScript MRR | Ruby MRR | Python MRR | PHP MRR | Go MRR | Avg. MRR | Avg. R@1 | Avg. R@5 | Avg. R@10 | Delta MRR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in rows.items():
        values = [pct(row["per_language"][lang]["MRR"]) for lang in LANGUAGES]
        delta = (
            "--"
            if name == "UA-HN source"
            else f"{100.0 * row['delta_mrr_vs_source']:+.2f}"
        )
        md.append(
            "| "
            + name
            + " | "
            + " | ".join(values)
            + f" | {pct(row['macro']['MRR'])} | {pct(row['macro']['R@1'])}"
            + f" | {pct(row['macro']['R@5'])} | {pct(row['macro']['R@10'])} | {delta} |"
        )
    md += [
        "",
        "## Reporting notes",
        "",
        (
            f"- FC refinement improves macro MRR from {pct(source_mrr)} to "
            f"{pct(rows['ReFCode-FC global']['macro']['MRR'])} "
            f"({100.0 * rows['ReFCode-FC global']['delta_mrr_vs_source']:+.2f} points)."
        ),
        (
            f"- Frozen parameter-free reranking reaches {pct(final_mrr)}: "
            f"{100.0 * (final_mrr - rows['ReFCode-FC global']['macro']['MRR']):+.2f} "
            "points over FC global and "
            f"{100.0 * (final_mrr - source_mrr):+.2f} points over the source."
        ),
        (
            f"- The older LI-trained fusion reaches {pct(legacy_li_mrr)} and is retained "
            "only as removal evidence; it does not outperform the frozen no-LI method."
        ),
        "- CoCoSoDa original-paper values, if shown, must be separated and labeled as reported rather than locally rerun.",
    ]
    md_path = OUT / "TABLE2_FINAL_NO_LI.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(OUT),
                "selected_k": selected_k,
                "selected_alpha": selected_alpha,
                "source_mrr": source_mrr,
                "fc_global_mrr": rows["ReFCode-FC global"]["macro"]["MRR"],
                "fc_fusion_mrr": final_mrr,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
