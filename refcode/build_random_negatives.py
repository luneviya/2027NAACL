"""Create row-aligned random negative candidate lists for the FC control study."""

import argparse
import json
import pickle
import random
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    urls = []
    with open(args.train_data_file, encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if line.strip():
                row = json.loads(line)
                urls.append(str(row.get("url", row.get("retrieval_idx", index))))

    if len(urls) < 2:
        raise ValueError("At least two training examples are required.")

    rng = random.Random(args.seed)
    candidates = []
    for index, url in enumerate(urls):
        choices = []
        while len(choices) < args.topk:
            candidate = rng.randrange(len(urls))
            if candidate == index or candidate in choices or urls[candidate] == url:
                continue
            choices.append(candidate)
        candidates.append(choices)

    output = Path(args.output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        pickle.dump(
            {
                "self_mined_idx": candidates,
                "candidate_source": "random",
                "seed": args.seed,
                "topk": args.topk,
                "train_data_file": args.train_data_file,
            },
            handle,
        )
    print(f"saved {len(candidates)} random candidate lists to {output}")


if __name__ == "__main__":
    main()
