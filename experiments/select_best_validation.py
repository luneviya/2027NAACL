#!/usr/bin/env python3
"""Select one completed variant by validation MRR without looking at test metrics."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--summary-file", default="")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    variants = []
    for metrics_file in sorted(root.glob("*/best_validation.json")):
        output_dir = metrics_file.parent
        checkpoint = output_dir / "checkpoint-best-mrr" / "model.bin"
        if not checkpoint.is_file():
            continue
        with metrics_file.open(encoding="utf-8") as handle:
            metrics = json.load(handle)
        variants.append(
            {
                "variant": output_dir.name,
                "output_dir": str(output_dir),
                "checkpoint": str(checkpoint),
                "selection_metric": metrics["selection_metric"],
                "selection_mrr": float(metrics["selection_mrr"]),
                "global_mrr": float(metrics["global_mrr"]),
                "fusion_mrr": float(metrics.get("fusion_mrr", 0.0)),
                "late_mrr": float(metrics.get("late_mrr", 0.0)),
                "epoch": int(metrics["epoch"]),
            }
        )

    if not variants:
        raise SystemExit(f"No completed variants found under {root}")

    variants.sort(key=lambda row: (-row["selection_mrr"], row["variant"]))
    selected = variants[0]
    summary = {
        "selection_rule": "maximum fixed-alpha validation fusion MRR; test is not inspected",
        "selected_variant": selected["variant"],
        "selected_dir": selected["output_dir"],
        "variants": variants,
    }
    summary_file = Path(args.summary_file) if args.summary_file else root / "selection_summary.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with summary_file.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(selected["output_dir"])


if __name__ == "__main__":
    main()
