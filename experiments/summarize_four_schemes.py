"""Collect validation-only A-D screening results into one JSON table."""

import argparse
import json
from pathlib import Path


def read_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    summary = {
        "protocol": {
            "split": "validation",
            "seed": 123456,
            "source": "UA-HN without CTRD",
            "test_set_used_for_selection": False,
        },
        "languages": {},
    }
    for language in ["javascript", "python"]:
        language_root = root / language
        a = read_json(language_root / "A_boundary_preserve" / "best_validation.json")
        b = read_json(language_root / "B_local_distillation" / "best_validation.json")
        structural = read_json(language_root / "structural" / "validation.json")
        global_a = float(a["global_mrr"])
        global_b = float(b["global_mrr"])
        structural_global = float(structural["global"]["mrr"])
        c = structural["schemes"]["difference_aware"]
        d = structural["schemes"]["compact_multivector"]
        rows = [
            {
                "scheme": "A_boundary_preserve",
                "inference": "single_vector",
                "validation_mrr": global_a,
                "matched_baseline_mrr": global_a,
                "delta_vs_matched_baseline": 0.0,
            },
            {
                "scheme": "B_local_distillation",
                "inference": "single_vector",
                "validation_mrr": global_b,
                "matched_baseline_mrr": global_a,
                "delta_vs_matched_baseline": global_b - global_a,
            },
            {
                "scheme": "C_candidate_difference",
                "inference": "top50_rerank",
                "validation_mrr": float(c["best_validation"]["mrr"]),
                "matched_baseline_mrr": structural_global,
                "delta_vs_matched_baseline": float(c["delta_mrr_vs_global"]),
                "selected_global_alpha": float(c["best_alpha"]),
            },
            {
                "scheme": "D_compact_multivector",
                "inference": "top50_multivector_rerank",
                "validation_mrr": float(d["best_validation"]["mrr"]),
                "matched_baseline_mrr": structural_global,
                "delta_vs_matched_baseline": float(d["delta_mrr_vs_global"]),
                "selected_global_alpha": float(d["best_alpha"]),
                "num_vectors": int(structural["num_vectors"]),
            },
        ]
        summary["languages"][language] = rows

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
