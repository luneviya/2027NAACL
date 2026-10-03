#!/usr/bin/env python3
"""Summarize the frozen Java/JavaScript UniXcoder transfer evaluation."""

import argparse
import json
from pathlib import Path


LANGUAGES = ["java", "javascript"]


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
    parser.add_argument("--run-root", default="runs/backbone_transfer_no_li/unixcoder")
    parser.add_argument("--output-root", default="experimental_results/12_unixcoder_transfer_no_li_seed123456")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    run_root = repo / args.run_root
    output_root = repo / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    rows = {}
    for lang in LANGUAGES:
        source = read(run_root / "evaluation" / lang / "source_test" / "grid_results.json")
        final = read(run_root / "evaluation" / lang / "final_test" / "grid_results.json")
        source_protocol = source["protocol"]
        final_protocol = final["protocol"]
        if source.get("missing_gold") != 0 or final.get("missing_gold") != 0:
            raise RuntimeError(f"missing gold code for {lang}")
        if source_protocol.get("split") != "test" or final_protocol.get("split") != "test":
            raise RuntimeError(f"non-test evaluation found for {lang}")
        if final_protocol.get("ks") != [20] or final_protocol.get("alphas") != [0.5]:
            raise RuntimeError(f"unfrozen rerank protocol for {lang}")
        source_mrr = source["global"]["MRR"]
        global_mrr = final["global"]["MRR"]
        fusion_mrr = final["fusion"]["20"]["0.5"]["MRR"]
        rows[lang] = {
            "source_mrr": source_mrr,
            "refcode_global_mrr": global_mrr,
            "refcode_fusion_mrr": fusion_mrr,
            "global_delta": global_mrr - source_mrr,
            "fusion_delta": fusion_mrr - source_mrr,
            "rerank_delta": fusion_mrr - global_mrr,
            "source_protocol": source_protocol,
            "final_protocol": final_protocol,
        }

    macro = {
        key: mean(rows[lang][key] for lang in LANGUAGES)
        for key in [
            "source_mrr",
            "refcode_global_mrr",
            "refcode_fusion_mrr",
            "global_delta",
            "fusion_delta",
            "rerank_delta",
        ]
    }
    exported = {
        "protocol": {
            "backbone": "UniXcoder",
            "languages": LANGUAGES,
            "seed": 123456,
            "source": "UA-HN without CTRD",
            "refinement": "FC without LI auxiliary training",
            "inference": "parameter-free MaxSim, frozen K=20 and alpha=0.5 from the main six-language validation sweep",
            "selection": "no transfer-specific test tuning",
        },
        "per_language": rows,
        "macro": macro,
    }
    (output_root / "unixcoder_transfer_no_li.json").write_text(json.dumps(exported, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# UniXcoder transfer: no CTRD, no LI auxiliary training",
        "",
        "K=20 and alpha=0.5 are copied unchanged from the main six-language validation selection; no transfer test score is used for tuning.",
        "",
        "| Language | Source MRR | ReFCode-G | ReFCode-F | Global delta | Fusion delta | Rerank delta |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for lang in LANGUAGES:
        row = rows[lang]
        lines.append(
            f"| {lang.title()} | {pct(row['source_mrr'])} | {pct(row['refcode_global_mrr'])} | "
            f"{pct(row['refcode_fusion_mrr'])} | {100 * row['global_delta']:+.2f} | "
            f"{100 * row['fusion_delta']:+.2f} | {100 * row['rerank_delta']:+.2f} |"
        )
    lines.append(
        f"| Macro avg. | {pct(macro['source_mrr'])} | {pct(macro['refcode_global_mrr'])} | "
        f"{pct(macro['refcode_fusion_mrr'])} | {100 * macro['global_delta']:+.2f} | "
        f"{100 * macro['fusion_delta']:+.2f} | {100 * macro['rerank_delta']:+.2f} |"
    )
    (output_root / "TABLE4B_UNIXCODER_TRANSFER.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"saved_to": str(output_root), "macro": macro}, indent=2))


if __name__ == "__main__":
    main()
