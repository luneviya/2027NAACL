#!/usr/bin/env python3
"""Export a paper-facing Table 3 copy for the no-LI-training positioning."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LANGUAGES = ["java", "javascript", "ruby", "python", "php", "go"]
METRICS = ["MRR", "R@1", "R@5", "R@10"]
OUT = ROOT / "experimental_results/06b_table3_no_li_positioning_seed123456"


def read(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def macro(per_language):
    return {
        metric: sum(per_language[lang][metric] for lang in LANGUAGES) / len(LANGUAGES)
        for metric in METRICS
    }


def pct(value):
    return f"{100.0 * value:.2f}"


def main():
    original = read(
        ROOT / "experimental_results/06_table3_noctrd_seed123456/table3_results.json"
    )
    table5 = read(
        ROOT / "experimental_results/07_table5_efficiency_no_li/table5_results.json"
    )
    table2 = read(
        ROOT
        / "experimental_results/08_final_no_li_main_seed123456/table2_final_no_li.json"
    )

    source = original["panel_a_candidate_source"]["Source (no refinement)"]
    panel_a = original["panel_a_candidate_source"]
    exact_source = table2["table2_rows"]["UA-HN source"]
    panel_a["Source (no refinement)"]["per_language"] = exact_source["per_language"]
    panel_a["Source (no refinement)"]["macro"] = exact_source["macro"]
    exact_source_mrr = exact_source["macro"]["MRR"]
    for row in panel_a.values():
        row["delta_mrr"] = row["macro"]["MRR"] - exact_source_mrr
    selected = table5["selection"]["best_configuration"]
    selected_k = int(selected["k"])
    selected_alpha = float(selected["alpha"])
    alpha_key = f"{selected_alpha:g}"
    no_li_global = {}
    no_li_rerank = {}
    for lang in LANGUAGES:
        result_file = (
            ROOT / "runs/table5_efficiency_no_li/test" / lang / "grid_results.json"
        )
        result = read(result_file)
        protocol = result["protocol"]
        if protocol["seed"] != 123456 or protocol["split"] != "test":
            raise RuntimeError(f"Unexpected protocol in {result_file}: {protocol}")
        if protocol["gold_url_coverage"] != 1.0 or protocol["empty_gold_code"] != 0:
            raise RuntimeError(f"Failed corpus preflight in {result_file}")
        if result["missing_gold"] != 0:
            raise RuntimeError(f"Missing gold entries in {result_file}")
        no_li_global[lang] = {
            metric: float(result["global"][metric]) for metric in METRICS
        }
        no_li_rerank[lang] = {
            metric: float(result["fusion"][str(selected_k)][alpha_key][metric])
            for metric in METRICS
        }

    panel_b = {
        "UA-HN source": {
            "training": "source retriever",
            "inference": "global retrieval",
            "per_language": exact_source["per_language"],
            "macro": exact_source["macro"],
        },
        "ReFCode-FC global": {
            "training": "failure-candidate refinement; LI auxiliary disabled",
            "inference": "global retrieval",
            "per_language": no_li_global,
            "macro": macro(no_li_global),
        },
        "ReFCode-FC + local rerank": {
            "training": "failure-candidate refinement; LI auxiliary disabled",
            "inference": (
                f"parameter-free token MaxSim reranking; K={selected_k}; "
                f"alpha={selected_alpha:g}"
            ),
            "per_language": no_li_rerank,
            "macro": macro(no_li_rerank),
            "status": "final; selected on six-language validation macro MRR",
        },
    }
    panel_b["ReFCode-FC global"]["macro"] = {
        metric: float(table5["global_test_macro"][metric]) for metric in METRICS
    }
    panel_b["ReFCode-FC + local rerank"]["macro"] = {
        metric: float(table5["fusion_rows"][str(selected_k)]["test_macro"][metric])
        for metric in METRICS
    }
    source_mrr = panel_b["UA-HN source"]["macro"]["MRR"]
    global_mrr = panel_b["ReFCode-FC global"]["macro"]["MRR"]
    rerank_mrr = panel_b["ReFCode-FC + local rerank"]["macro"]["MRR"]
    for row in panel_b.values():
        row["delta_mrr_vs_source"] = row["macro"]["MRR"] - source_mrr

    li_trained_fusion = float(
        original["panel_b_components"]["FC, LI on, fusion"]["macro"]["MRR"]
    )
    exported = {
        "positioning": {
            "paper_model": "ReFCode without LI auxiliary training",
            "training": "failure-candidate refinement of the global retriever",
            "inference": "global top-K retrieval followed by parameter-free local-evidence reranking",
            "terminology": "training uses labeled retrieval failures; inference reranks an unlabeled candidate neighborhood",
        },
        "protocol": {
            "seed": 123456,
            "ctrd_used": False,
            "li_auxiliary_training": False,
            "languages": LANGUAGES,
            "current_rerank_setting": (
                f"K={selected_k}, alpha={selected_alpha:g}; frozen by six-language "
                "validation macro MRR"
            ),
            "validation_macro_mrr": float(selected["validation_macro_mrr"]),
            "statistical_scope": "single fixed seed; no multi-seed significance claim",
        },
        "panel_a_candidate_source": panel_a,
        "panel_b_no_li_method": panel_b,
        "discarded_li_training_audit": {
            "no_li_fusion_macro_mrr": rerank_mrr,
            "li_trained_fusion_macro_mrr": li_trained_fusion,
            "li_training_delta_mrr": li_trained_fusion - rerank_mrr,
            "interpretation": (
                "The older LI-trained fusion does not outperform the final frozen "
                "no-LI method and is retained only as removal evidence."
            ),
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "table3_no_li_positioning.json").write_text(
        json.dumps(exported, indent=2) + "\n", encoding="utf-8"
    )

    md = [
        "# Table 3 copy: no-LI-training paper positioning",
        "",
        "This copy does not overwrite the original Table 3 export. All values are six-language macro percentages under seed 123456.",
        "",
        "## Panel A: candidate-source control (equal-budget K=1)",
        "",
        "| Candidate source | MRR | R@1 | R@5 | R@10 | Delta MRR vs. source |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in panel_a.items():
        m = row["macro"]
        delta = "--" if name == "Source (no refinement)" else f"{100.0 * row['delta_mrr']:+.2f}"
        md.append(
            f"| {name} | {pct(m['MRR'])} | {pct(m['R@1'])} | {pct(m['R@5'])} | {pct(m['R@10'])} | {delta} |"
        )

    md += [
        "",
        "## Panel B: final no-LI-training method",
        "",
        "| System | Training | Inference | MRR | R@1 | R@5 | R@10 | Delta MRR vs. source |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in panel_b.items():
        m = row["macro"]
        delta = "--" if name == "UA-HN source" else f"{100.0 * row['delta_mrr_vs_source']:+.2f}"
        md.append(
            f"| {name} | {row['training']} | {row['inference']} | {pct(m['MRR'])} | "
            f"{pct(m['R@1'])} | {pct(m['R@5'])} | {pct(m['R@10'])} | {delta} |"
        )

    md += [
        "",
        "## Per-language MRR for the final method",
        "",
        "| System | " + " | ".join(LANGUAGES) + " | Macro |",
        "|---|" + "---:|" * 7,
    ]
    for name, row in panel_b.items():
        values = [pct(row["per_language"][lang]["MRR"]) for lang in LANGUAGES]
        md.append("| " + name + " | " + " | ".join(values) + f" | {pct(row['macro']['MRR'])} |")

    md += [
        "",
        "## Paper-facing interpretation",
        "",
        f"- Failure-candidate refinement improves global MRR from {pct(source_mrr)} to {pct(global_mrr)} ({100.0 * (global_mrr - source_mrr):+.2f} points).",
        f"- Parameter-free local reranking improves it further to {pct(rerank_mrr)} ({100.0 * (rerank_mrr - global_mrr):+.2f} points over the global branch).",
        f"- The older LI-trained fusion reaches {pct(li_trained_fusion)} versus {pct(rerank_mrr)} for the final no-LI method ({100.0 * (li_trained_fusion - rerank_mrr):+.3f} points); it is retained only as removal evidence.",
        f"- The final reranker setting is K={selected_k}, alpha={selected_alpha:g}, selected by six-language validation macro MRR and frozen before test reporting.",
        "- Describe inference as reranking an unlabeled candidate neighborhood, not identifying failures with gold labels.",
    ]
    (OUT / "TABLE3_NO_LI_POSITIONING.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(OUT),
        "source_mrr": source_mrr,
        "fc_global_mrr": global_mrr,
        "fc_rerank_mrr": rerank_mrr,
        "li_training_delta_mrr": li_trained_fusion - rerank_mrr,
    }, indent=2))


if __name__ == "__main__":
    main()
