"""Validation-only screening for candidate-difference and compact-vector scoring."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

from refcode.run_refcode_refinement import (
    RerankCodeSearchDataset,
    load_rerank_model,
    rerank_collate_ids,
    rerank_encode_pooled,
    rerank_get_last_hidden,
    rerank_mrr_from_ranks,
    rerank_recall_from_ranks,
    rerank_token_mask,
    rerank_zscore,
)


def compact_multivectors(hidden, mask, num_vectors):
    """Mean-pool contiguous valid-token regions into a few normalized vectors."""
    batch, _, dim = hidden.shape
    pooled = hidden.new_zeros((batch, num_vectors, dim))
    for row in range(batch):
        valid_idx = mask[row].nonzero(as_tuple=False).squeeze(1)
        if valid_idx.numel() == 0:
            continue
        fallback = hidden[row].index_select(0, valid_idx).mean(dim=0)
        chunks = torch.tensor_split(valid_idx, num_vectors)
        for part, indices in enumerate(chunks):
            pooled[row, part] = (
                hidden[row].index_select(0, indices).mean(dim=0)
                if indices.numel() > 0 else fallback
            )
    return F.normalize(pooled, p=2, dim=-1)


def metrics_from_ranks(ranks):
    return {
        "mrr": rerank_mrr_from_ranks(ranks),
        "r1": rerank_recall_from_ranks(ranks, 1),
        "r5": rerank_recall_from_ranks(ranks, 5),
        "r10": rerank_recall_from_ranks(ranks, 10),
        "r50": rerank_recall_from_ranks(ranks, 50),
    }


def rank_in_candidates(candidate_ids, candidate_scores, gold_idx):
    order = np.argsort(candidate_scores)[::-1]
    ranked_ids = candidate_ids[order]
    return int(np.where(ranked_ids == gold_idx)[0][0]) + 1


def evaluate(args):
    started = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    args.fp16 = bool(args.fp16)
    model, tokenizer = load_rerank_model(args)

    query_dataset = RerankCodeSearchDataset(
        args.eval_data_file, tokenizer, args.code_length, args.nl_length, "query"
    )
    code_dataset = RerankCodeSearchDataset(
        args.codebase_file, tokenizer, args.code_length, args.nl_length, "code"
    )
    query_loader = DataLoader(
        query_dataset,
        sampler=SequentialSampler(query_dataset),
        batch_size=args.eval_batch_size,
        collate_fn=rerank_collate_ids,
        num_workers=args.num_workers,
    )
    code_loader = DataLoader(
        code_dataset,
        sampler=SequentialSampler(code_dataset),
        batch_size=args.eval_batch_size,
        collate_fn=rerank_collate_ids,
        num_workers=args.num_workers,
    )
    query_vecs = rerank_encode_pooled(
        model, query_loader, device, is_code=False, fp16=args.fp16
    )
    code_vecs = rerank_encode_pooled(
        model, code_loader, device, is_code=True, fp16=args.fp16
    )

    code_urls = [example["url"] for example in code_dataset.examples]
    query_urls = [example["url"] for example in query_dataset.examples]
    url_to_code_idx = {}
    for idx, url in enumerate(code_urls):
        url_to_code_idx.setdefault(url, idx)

    alphas = [float(value) for value in args.alphas.split(",")]
    bi_ranks = []
    standalone_ranks = {"difference_aware": [], "compact_multivector": []}
    fused_ranks = {
        scheme: {alpha: [] for alpha in alphas}
        for scheme in standalone_ranks
    }
    code_vecs_device = code_vecs.to(device)
    query_vecs_device = query_vecs.to(device)

    iterator = range(len(query_dataset))
    for query_idx in tqdm(iterator, desc=f"Structural rerank top{args.top_k}"):
        pooled_scores = torch.matmul(
            query_vecs_device[query_idx: query_idx + 1], code_vecs_device.t()
        ).squeeze(0)
        gold_idx = url_to_code_idx.get(query_urls[query_idx])
        if gold_idx is None:
            bi_ranks.append(0)
            for scheme in standalone_ranks:
                standalone_ranks[scheme].append(0)
                for alpha in alphas:
                    fused_ranks[scheme][alpha].append(0)
            continue

        gold_score = pooled_scores[gold_idx]
        bi_rank = int((pooled_scores > gold_score).sum().item()) + 1
        bi_ranks.append(bi_rank)
        top_k = min(args.top_k, pooled_scores.numel())
        top_scores, top_indices = torch.topk(pooled_scores, k=top_k)
        top_ids = top_indices.detach().cpu().numpy()
        pooled_top = top_scores.detach().float().cpu().numpy()

        if gold_idx not in set(top_ids.tolist()):
            for scheme in standalone_ranks:
                standalone_ranks[scheme].append(bi_rank)
                for alpha in alphas:
                    fused_ranks[scheme][alpha].append(bi_rank)
            continue

        query_ids = torch.tensor(
            query_dataset.examples[query_idx]["nl_ids"],
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=args.fp16 and device.type == "cuda"):
                query_hidden = rerank_get_last_hidden(model, query_ids)
            query_mask = rerank_token_mask(query_ids, tokenizer)
        query_hidden = query_hidden.float()
        query_normalized = F.normalize(query_hidden[0], p=2, dim=-1)
        query_multi = compact_multivectors(
            query_hidden, query_mask, args.num_vectors
        )[0]

        token_match_chunks = []
        multivector_chunks = []
        for start in range(0, top_k, args.rerank_batch_size):
            end = min(start + args.rerank_batch_size, top_k)
            candidate_ids = [
                code_dataset.examples[int(code_idx)]["code_ids"]
                for code_idx in top_ids[start:end]
            ]
            code_ids = torch.tensor(candidate_ids, dtype=torch.long, device=device)
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=args.fp16 and device.type == "cuda"):
                    code_hidden = rerank_get_last_hidden(model, code_ids)
                code_mask = rerank_token_mask(code_ids, tokenizer)
            code_hidden = code_hidden.float()

            code_normalized = F.normalize(code_hidden, p=2, dim=-1)
            token_similarity = torch.einsum(
                "qd,bkd->bqk", query_normalized, code_normalized
            )
            token_similarity = token_similarity.masked_fill(
                ~code_mask.unsqueeze(1), -1e4
            )
            token_match_chunks.append(
                token_similarity.max(dim=2).values.detach().cpu()
            )

            code_multi = compact_multivectors(
                code_hidden, code_mask, args.num_vectors
            )
            multi_similarity = torch.einsum(
                "md,bnd->bmn", query_multi, code_multi
            )
            multivector_chunks.append(
                multi_similarity.max(dim=2).values.mean(dim=1).detach().cpu()
            )

        token_matches = torch.cat(token_match_chunks, dim=0)
        valid_query = query_mask[0].detach().cpu()
        valid_matches = token_matches[:, valid_query]
        dispersion = valid_matches.std(dim=0, unbiased=False).clamp_min(1e-4)
        difference_weights = dispersion.pow(args.difference_power)
        difference_scores = (
            (valid_matches * difference_weights.unsqueeze(0)).sum(dim=1)
            / difference_weights.sum().clamp_min(1e-6)
        ).numpy()
        multivector_scores = torch.cat(multivector_chunks, dim=0).numpy()
        structural_scores = {
            "difference_aware": difference_scores,
            "compact_multivector": multivector_scores,
        }

        for scheme, scheme_scores in structural_scores.items():
            standalone_ranks[scheme].append(
                rank_in_candidates(top_ids, scheme_scores, gold_idx)
            )
            for alpha in alphas:
                final_scores = (
                    alpha * rerank_zscore(pooled_top)
                    + (1.0 - alpha) * rerank_zscore(scheme_scores)
                )
                fused_ranks[scheme][alpha].append(
                    rank_in_candidates(top_ids, final_scores, gold_idx)
                )

    global_metrics = metrics_from_ranks(bi_ranks)
    schemes = {}
    for scheme in standalone_ranks:
        standalone = metrics_from_ranks(standalone_ranks[scheme])
        fusion = {
            f"{alpha:g}": metrics_from_ranks(fused_ranks[scheme][alpha])
            for alpha in alphas
        }
        best_alpha = max(alphas, key=lambda alpha: fusion[f"{alpha:g}"]["mrr"])
        best_metrics = fusion[f"{best_alpha:g}"]
        schemes[scheme] = {
            "standalone": standalone,
            "fusion": fusion,
            "best_alpha": best_alpha,
            "best_validation": best_metrics,
            "delta_mrr_vs_global": best_metrics["mrr"] - global_metrics["mrr"],
        }

    result = {
        "protocol": "validation_only",
        "checkpoint": args.loaded_model_filename,
        "eval_data_file": args.eval_data_file,
        "codebase_file": args.codebase_file,
        "num_queries": len(query_dataset),
        "num_codes": len(code_dataset),
        "top_k": args.top_k,
        "num_vectors": args.num_vectors,
        "difference_power": args.difference_power,
        "global": global_metrics,
        "schemes": schemes,
        "elapsed_seconds": time.time() - started,
    }
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"saved_to: {output_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loaded-model-filename", required=True)
    parser.add_argument("--eval-data-file", required=True)
    parser.add_argument("--codebase-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--model-name-or-path", default="DeepSoftwareAnalytics/CoCoSoDa")
    parser.add_argument("--config-name", default="DeepSoftwareAnalytics/CoCoSoDa")
    parser.add_argument("--tokenizer-name", default="DeepSoftwareAnalytics/CoCoSoDa")
    parser.add_argument("--code-length", type=int, default=256)
    parser.add_argument("--nl-length", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--rerank-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--num-vectors", type=int, default=4)
    parser.add_argument("--difference-power", type=float, default=1.0)
    parser.add_argument("--alphas", default="0.25,0.5,0.75")
    parser.add_argument("--fp16", type=int, default=1)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
