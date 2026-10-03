"""Attach source-score boundary metadata to an existing FC candidate pool."""

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import torch

from refcode.run_failure_harvesting import (
    encode_ids,
    load_model,
    load_train_items,
    set_seed,
)


logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data-file", required=True)
    parser.add_argument("--candidate-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--checkpoint-file", required=True)
    parser.add_argument("--model-name-or-path", default="DeepSoftwareAnalytics/CoCoSoDa")
    parser.add_argument("--config-name", default="DeepSoftwareAnalytics/CoCoSoDa")
    parser.add_argument("--tokenizer-name", default="DeepSoftwareAnalytics/CoCoSoDa")
    parser.add_argument("--nl-length", type=int, default=128)
    parser.add_argument("--code-length", type=int, default=256)
    parser.add_argument("--encode-batch-size", type=int, default=128)
    parser.add_argument("--score-batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123456)
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.candidate_file, "rb") as handle:
        candidate_obj = pickle.load(handle)
    if isinstance(candidate_obj, dict):
        candidate_rows = (
            candidate_obj.get("self_mined_idx")
            or candidate_obj.get("mined_idx")
            or candidate_obj.get("hard_idx")
            or candidate_obj.get("hard_indices")
        )
        output = dict(candidate_obj)
    else:
        candidate_rows = candidate_obj
        output = {"self_mined_idx": candidate_obj}
    if candidate_rows is None:
        raise ValueError("Candidate file has no self-mined index rows.")

    # Reuse the same loading and tokenization path as failure harvesting.
    model_args = argparse.Namespace(
        checkpoint_file=args.checkpoint_file,
        output_dir="",
        model_name_or_path=args.model_name_or_path,
        config_name=args.config_name,
        tokenizer_name=args.tokenizer_name,
    )
    model, tokenizer = load_model(model_args, device)
    items = load_train_items(
        args.train_data_file,
        tokenizer,
        args.nl_length,
        args.code_length,
    )
    if len(candidate_rows) < len(items):
        raise ValueError(
            f"Candidate rows ({len(candidate_rows)}) are fewer than train rows ({len(items)})."
        )

    query_vecs = encode_ids(
        model,
        [item["nl_ids"] for item in items],
        args.encode_batch_size,
        device,
        mode="nl",
        num_workers=args.num_workers,
    )
    code_vecs = encode_ids(
        model,
        [item["code_ids"] for item in items],
        args.encode_batch_size,
        device,
        mode="code",
        num_workers=args.num_workers,
    )

    rows = [list(map(int, row)) for row in candidate_rows[: len(items)]]
    max_k = max(len(row) for row in rows)
    if min(len(row) for row in rows) != max_k:
        raise ValueError("All candidate rows must have the same length.")

    candidate_indices = torch.tensor(rows, dtype=torch.long)
    gold_scores = np.empty(len(items), dtype=np.float32)
    candidate_scores = np.empty((len(items), max_k), dtype=np.float32)

    for start in range(0, len(items), args.score_batch_size):
        end = min(start + args.score_batch_size, len(items))
        query = query_vecs[start:end]
        gold = code_vecs[start:end]
        indices = candidate_indices[start:end]
        candidates = code_vecs.index_select(0, indices.reshape(-1)).view(
            end - start, max_k, -1
        )
        gold_scores[start:end] = (query * gold).sum(dim=1).numpy()
        candidate_scores[start:end] = torch.einsum(
            "bd,bkd->bk", query, candidates
        ).numpy()

    gold_ranks = 1 + (candidate_scores > gold_scores[:, None]).sum(axis=1)
    top_margin = gold_scores - candidate_scores[:, 0]
    output.update(
        {
            "source_gold_scores": gold_scores,
            "source_candidate_scores": candidate_scores,
            "source_gold_ranks_within_pool": gold_ranks.astype(np.int16),
            "source_top_margin": top_margin.astype(np.float32),
            "boundary_checkpoint": args.checkpoint_file,
        }
    )

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info("Saved boundary metadata to %s", output_path)
    logger.info("True-failure rate: %.4f", float((gold_ranks > 1).mean()))
    logger.info(
        "Top-margin quantiles: %s",
        np.quantile(top_margin, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]).tolist(),
    )


if __name__ == "__main__":
    main()
