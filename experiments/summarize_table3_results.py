#!/usr/bin/env python3
"""Summarize the completed seed-123456 no-CTRD Table 3 experiments."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LANGUAGES = ["java", "javascript", "ruby", "python", "php", "go"]
METRICS = ["MRR", "R@1", "R@5", "R@10"]
OUT = ROOT / "experimental_results/06_table3_noctrd_seed123456"


def read_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def method_result(path, method):
    rows = [row for row in read_jsonl(path) if row["method"] == method]
    if len(rows) != 1:
        raise RuntimeError(f"Expected one {method!r} row in {path}; found {len(rows)}")
    return {metric: float(rows[0][metric]) for metric in METRICS}


def macro(per_language):
    return {
        metric: sum(per_language[lang][metric] for lang in LANGUAGES) / len(LANGUAGES)
        for metric in METRICS
    }


def pct(value):
    return f"{100 * value:.2f}"


def signed_pct(value):
    return f"{100 * value:+.2f}"


def validate_li_off(path):
    provenance = path.read_text(encoding="utf-8")
    for marker in ["seed=123456", "use_lite_late_interaction_train=0", "li_weight=0"]:
        if marker not in provenance:
            raise RuntimeError(f"Missing {marker!r} in {path}")


def main():
    main_path = ROOT / "experimental_results/05_refcode_noctrd_main_seed123456/main_results.json"
    main_results = read_json(main_path)
    source = {
        lang: {
            "MRR": main_results["languages"][lang]["source"]["mrr"],
            "R@1": main_results["languages"][lang]["source"]["r1"],
            "R@5": main_results["languages"][lang]["source"]["r5"],
            "R@10": main_results["languages"][lang]["source"]["r10"],
        }
        for lang in LANGUAGES
    }

    panel_a = {"Source (no refinement)": {"per_language": source}}
    candidate_root = ROOT / "runs/table3_controls/candidate_source/ua_hn"
    for label, condition in [("Random", "random"), ("Static", "static"), ("ReFCode-FC", "fc")]:
        per_language = {}
        for lang in LANGUAGES:
            run_dir = candidate_root / condition / "seed_123456" / lang
            validate_li_off(run_dir / "provenance.txt")
            per_language[lang] = method_result(run_dir / "result.jsonl", "bi_encoder")
        panel_a[label] = {"per_language": per_language}
    for row in panel_a.values():
        row["macro"] = macro(row["per_language"])

    no_li_root = ROOT / "runs/table3_controls/components/fc_no_li/ua_hn/fc/seed_123456"
    no_li = {}
    for lang in LANGUAGES:
        validate_li_off(no_li_root / lang / "provenance.txt")
        no_li[lang] = method_result(no_li_root / lang / "result.jsonl", "bi_encoder")

    audit_root = ROOT / "experimental_results/05_refcode_noctrd_main_seed123456"
    audited = {}
    for lang in LANGUAGES:
        audit_file = audit_root / lang / "audited_top50_alpha0.5/result.jsonl"
        audited[lang] = {
            method: method_result(audit_file, method)
            for method in ["bi_encoder", "late_interaction", "fusion"]
        }

    main_key = {
        "bi_encoder": "refcode_global",
        "late_interaction": "refcode_local",
        "fusion": "refcode_fusion",
    }
    panel_b = {"FC, LI off, global": {"per_language": no_li}}
    for label, method in [
        ("FC, LI on, global", "bi_encoder"),
        ("FC, LI on, LI-only", "late_interaction"),
        ("FC, LI on, fusion", "fusion"),
    ]:
        per_language = {}
        for lang in LANGUAGES:
            row = dict(audited[lang][method])
            # Preserve the full-precision MRR from the previously exported main result.
            row["MRR"] = float(main_results["languages"][lang][main_key[method]]["mrr"])
            per_language[lang] = row
        panel_b[label] = {"per_language": per_language}
    for row in panel_b.values():
        row["macro"] = macro(row["per_language"])

    baseline_a = panel_a["Source (no refinement)"]["macro"]["MRR"]
    for row in panel_a.values():
        row["delta_mrr"] = row["macro"]["MRR"] - baseline_a
    baseline_b = panel_b["FC, LI off, global"]["macro"]["MRR"]
    for row in panel_b.values():
        row["delta_mrr"] = row["macro"]["MRR"] - baseline_b

    exported = {
        "protocol": {
            "seed": 123456,
            "languages": LANGUAGES,
            "source": "CoCoSoDa + UA-HN, no CTRD",
            "candidate_source_panel": "equal-budget K=1; LI training disabled; bi-encoder output only",
            "component_panel": "normal FC budget; top-50 fusion with alpha=0.5 for the full model",
            "statistical_scope": "single fixed seed; report paired language-wise effects, not significance claims",
        },
        "panel_a_candidate_source": panel_a,
        "panel_b_components": panel_b,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "table3_results.json").write_text(json.dumps(exported, indent=2) + "\n", encoding="utf-8")

    md = [
        "# Table 3 results: no-CTRD ReFCode, seed 123456",
        "",
        "All values below are percentages. The delta column is the absolute MRR-point change.",
        "",
        "## Panel A: candidate-source control (equal-budget K=1)",
        "",
        "| Candidate source | MRR | R@1 | R@5 | R@10 | Δ MRR vs. source |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, row in panel_a.items():
        m = row["macro"]
        delta = "--" if label == "Source (no refinement)" else signed_pct(row["delta_mrr"])
        md.append(f"| {label} | {pct(m['MRR'])} | {pct(m['R@1'])} | {pct(m['R@5'])} | {pct(m['R@10'])} | {delta} |")

    md += [
        "",
        "## Panel B: component contribution (normal FC budget)",
        "",
        "| Training / inference variant | MRR | R@1 | R@5 | R@10 | Δ MRR vs. FC, LI off |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, row in panel_b.items():
        m = row["macro"]
        delta = "--" if label == "FC, LI off, global" else signed_pct(row["delta_mrr"])
        md.append(f"| {label} | {pct(m['MRR'])} | {pct(m['R@1'])} | {pct(m['R@5'])} | {pct(m['R@10'])} | {delta} |")

    for title, panel in [("Panel A per-language MRR", panel_a), ("Panel B per-language MRR", panel_b)]:
        md += ["", f"## {title}", "", "| Variant | " + " | ".join(LANGUAGES) + " | Macro |", "|---|" + "---:|" * 7]
        for label, row in panel.items():
            values = [pct(row["per_language"][lang]["MRR"]) for lang in LANGUAGES]
            md.append("| " + label + " | " + " | ".join(values) + f" | {pct(row['macro']['MRR'])} |")

    md += [
        "",
        "## Reporting notes",
        "",
        "- Random and static controls do not improve over the source retriever.",
        "- Equal-budget ReFCode-FC raises macro MRR by 0.35 points and R@1 by 0.82 points, while R@10 falls by 0.52 points; its clearest effect is top-rank calibration rather than broader recall.",
        "- Turning on LI training changes global MRR by only +0.04 points relative to FC without LI.",
        "- Full fusion reaches 81.05 MRR: +0.48 points over FC without LI, +0.44 over the LI-trained global branch, and +1.28 over the unreﬁned source retriever.",
        "- These are one-seed results (123456). Use cross-language consistency and effect sizes; do not make a p-value or multi-seed stability claim.",
        "- LI-off runs emitted LI/fusion diagnostic rows, but those rows are intentionally excluded because the local branch was not trained.",
    ]
    (OUT / "TABLE3_RESULTS.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(json.dumps({
        "panel_a": {name: row["macro"] for name, row in panel_a.items()},
        "panel_b": {name: row["macro"] for name, row in panel_b.items()},
        "output": str(OUT),
    }, indent=2))


if __name__ == "__main__":
    main()
