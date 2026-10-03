#!/usr/bin/env python3
"""One-pass ReFCode K/alpha grid with online-latency and VRAM profiling.

Unlike the legacy sweep shell script, this program encodes each split once,
computes late-interaction scores up to max(K) once, and derives every smaller-K
and alpha result from that same run.  It also reranks every query so that the
latency measurement does not depend on whether the evaluator knows the gold
candidate is inside top-K.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from refcode import run_refcode_refinement as rr


def csv_ints(value):
    return sorted({int(item) for item in value.split(",") if item.strip()})


def csv_floats(value):
    return sorted({float(item) for item in value.split(",") if item.strip()})


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def metrics(ranks):
    return {
        "MRR": rr.rerank_mrr_from_ranks(ranks),
        "R@1": rr.rerank_recall_from_ranks(ranks, 1),
        "R@5": rr.rerank_recall_from_ranks(ranks, 5),
        "R@10": rr.rerank_recall_from_ranks(ranks, 10),
        "R@50": rr.rerank_recall_from_ranks(ranks, 50),
    }


def alpha_key(value):
    return f"{value:g}"


def timed_pooled_encode(model, loader, device, is_code, fp16):
    sync(device)
    started = time.perf_counter()
    vectors = rr.rerank_encode_pooled(model, loader, device, is_code=is_code, fp16=fp16)
    sync(device)
    return vectors, time.perf_counter() - started


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["valid", "test"], required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ks", default="10,20,50,100")
    parser.add_argument("--alphas", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--base-model", default="DeepSoftwareAnalytics/CoCoSoDa")
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--rerank-batch-size", type=int, default=64)
    parser.add_argument("--code-length", type=int, default=256)
    parser.add_argument("--nl-length", type=int, default=128)
    parser.add_argument("--fp16", type=int, default=1)
    parser.add_argument(
        "--global-only",
        action="store_true",
        help="evaluate pooled global retrieval only; skip token-level reranking",
    )
    parser.add_argument(
        "--save-per-query",
        action="store_true",
        help="write query_ranks.jsonl beside the aggregate grid result",
    )
    args = parser.parse_args()

    ks = csv_ints(args.ks)
    alphas = csv_floats(args.alphas)
    if not ks or not alphas:
        raise ValueError("--ks and --alphas must be non-empty")
    if min(ks) <= 0 or min(alphas) < 0.0 or max(alphas) > 1.0:
        raise ValueError("K must be positive and alpha must lie in [0, 1]")

    root = ROOT
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    query_file = root / "dataset" / args.lang / f"{args.split}.jsonl"
    # Both valid and test URL lists are evaluated against the shared codebase.
    # The valid JSONL contains queries but intentionally has empty code fields.
    code_file = root / "dataset" / args.lang / "codebase.jsonl"
    for required in [query_file, code_file]:
        if not required.is_file():
            raise FileNotFoundError(required)

    # Fail before loading the model if the retrieval corpus cannot provide a
    # non-empty gold implementation for every query.
    query_rows = rr.read_rerank_jsonl(str(query_file))
    code_rows = rr.read_rerank_jsonl(str(code_file))
    code_by_url = {}
    for row in code_rows:
        url = row.get("url", row.get("retrieval_idx", ""))
        if url and (
            url not in code_by_url
            or not rr._rerank_extract_code_tokens(code_by_url[url])
        ):
            code_by_url[url] = row
    query_urls_preflight = [
        row.get("url", row.get("retrieval_idx", "")) for row in query_rows
    ]
    missing_urls = [url for url in query_urls_preflight if url not in code_by_url]
    empty_gold_code = [
        url
        for url in query_urls_preflight
        if url in code_by_url and not rr._rerank_extract_code_tokens(code_by_url[url])
    ]
    if missing_urls or empty_gold_code:
        raise RuntimeError(
            f"Invalid retrieval corpus: missing_gold={len(missing_urls)}, "
            f"empty_gold_code={len(empty_gold_code)}"
        )
    print(
        f"[Preflight] split={args.split} queries={len(query_rows)} "
        f"codes={len(code_rows)} gold_coverage=100% empty_gold_code=0"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fp16 = bool(args.fp16) and device.type == "cuda"
    torch.manual_seed(123456)
    np.random.seed(123456)

    model_args = SimpleNamespace(
        config_name=args.base_model,
        model_name_or_path=args.base_model,
        tokenizer_name=args.base_model,
        loaded_model_filename=str(checkpoint),
        device=device,
    )
    model, tokenizer = rr.load_rerank_model(model_args)
    query_dataset = rr.RerankCodeSearchDataset(
        str(query_file), tokenizer, args.code_length, args.nl_length, "query"
    )
    code_dataset = rr.RerankCodeSearchDataset(
        str(code_file), tokenizer, args.code_length, args.nl_length, "code"
    )
    query_loader = DataLoader(
        query_dataset,
        sampler=SequentialSampler(query_dataset),
        batch_size=args.eval_batch_size,
        collate_fn=rr.rerank_collate_ids,
        num_workers=4,
    )
    code_loader = DataLoader(
        code_dataset,
        sampler=SequentialSampler(code_dataset),
        batch_size=args.eval_batch_size,
        collate_fn=rr.rerank_collate_ids,
        num_workers=4,
    )

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    query_vecs, query_encode_seconds = timed_pooled_encode(
        model, query_loader, device, is_code=False, fp16=fp16
    )
    code_vecs, code_encode_seconds = timed_pooled_encode(
        model, code_loader, device, is_code=True, fp16=fp16
    )
    query_vecs = query_vecs.to(device)
    code_vecs = code_vecs.to(device)

    code_urls = [example["url"] for example in code_dataset.examples]
    query_urls = [example["url"] for example in query_dataset.examples]
    url_to_code_idx = {}
    for index, url in enumerate(code_urls):
        url_to_code_idx.setdefault(url, index)

    max_k = min(max(ks), len(code_dataset))
    bi_ranks = []
    li_ranks = {k: [] for k in ks}
    fusion_ranks = {k: {alpha_key(a): [] for a in alphas} for k in ks}
    global_search_seconds = 0.0
    rerank_seconds = 0.0
    missing_gold = 0

    for qi in tqdm(range(len(query_dataset)), desc=f"{args.lang}/{args.split} top{max_k} grid"):
        sync(device)
        started = time.perf_counter()
        scores = torch.matmul(query_vecs[qi : qi + 1], code_vecs.t()).squeeze(0)
        if args.global_only:
            top_scores = top_idx = None
        else:
            top_scores, top_idx = torch.topk(scores, k=max_k, dim=0)
        gold_idx = url_to_code_idx.get(query_urls[qi])
        if gold_idx is None:
            bi_rank = 0
            missing_gold += 1
        else:
            gold_score = scores[gold_idx]
            bi_rank = int((scores > gold_score).sum().item()) + 1
        sync(device)
        global_search_seconds += time.perf_counter() - started
        bi_ranks.append(bi_rank)

        if args.global_only:
            continue

        top_idx_np = top_idx.detach().cpu().numpy()
        bi_scores_np = top_scores.detach().float().cpu().numpy()

        sync(device)
        started = time.perf_counter()
        q_ids = torch.tensor(
            query_dataset.examples[qi]["nl_ids"], dtype=torch.long, device=device
        ).unsqueeze(0)
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=fp16):
                q_hidden = rr.rerank_get_last_hidden(model, q_ids)
            q_mask = rr.rerank_token_mask(q_ids, tokenizer)
            q_hidden = q_hidden.float().expand(max_k, -1, -1)
            q_mask = q_mask.expand(max_k, -1)

        li_chunks = []
        for begin in range(0, max_k, args.rerank_batch_size):
            end = min(begin + args.rerank_batch_size, max_k)
            candidate_ids = [
                code_dataset.examples[int(cid)]["code_ids"] for cid in top_idx_np[begin:end]
            ]
            c_ids = torch.tensor(candidate_ids, dtype=torch.long, device=device)
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=fp16):
                    c_hidden = rr.rerank_get_last_hidden(model, c_ids)
                c_mask = rr.rerank_token_mask(c_ids, tokenizer)
                li_score = rr.rerank_late_interaction_score(
                    q_hidden[begin:end], q_mask[begin:end], c_hidden.float(), c_mask
                )
            li_chunks.append(li_score.detach().cpu())
        li_scores_np = torch.cat(li_chunks).numpy()
        sync(device)
        rerank_seconds += time.perf_counter() - started

        for requested_k in ks:
            k = min(requested_k, max_k)
            candidate_ids = top_idx_np[:k]
            if gold_idx is None or gold_idx not in candidate_ids:
                li_ranks[requested_k].append(bi_rank)
                for a in alphas:
                    fusion_ranks[requested_k][alpha_key(a)].append(bi_rank)
                continue

            local_scores = li_scores_np[:k]
            local_order = np.argsort(local_scores)[::-1]
            local_ranked_ids = candidate_ids[local_order]
            li_position = int(np.where(local_ranked_ids == gold_idx)[0][0]) + 1
            li_ranks[requested_k].append(li_position)

            global_scores = bi_scores_np[:k]
            global_z = rr.rerank_zscore(global_scores)
            local_z = rr.rerank_zscore(local_scores)
            for a in alphas:
                combined = a * global_z + (1.0 - a) * local_z
                combined_order = np.argsort(combined)[::-1]
                combined_ids = candidate_ids[combined_order]
                position = int(np.where(combined_ids == gold_idx)[0][0]) + 1
                fusion_ranks[requested_k][alpha_key(a)].append(position)

    n_queries = len(query_dataset)
    result = {
        "protocol": {
            "language": args.lang,
            "split": args.split,
            "checkpoint": str(checkpoint),
            "base_model": args.base_model,
            "query_file": str(query_file),
            "code_file": str(code_file),
            "gold_url_coverage": 1.0,
            "empty_gold_code": 0,
            "ks": ks,
            "alphas": alphas,
            "seed": 123456,
            "fp16": fp16,
            "eval_batch_size": args.eval_batch_size,
            "rerank_batch_size": args.rerank_batch_size,
            "global_only": args.global_only,
            "save_per_query": args.save_per_query,
            "latency_policy": (
                "global retrieval only"
                if args.global_only
                else "all queries reranked; no gold-aware compute skipping"
            ),
        },
        "num_queries": n_queries,
        "num_codes": len(code_dataset),
        "missing_gold": missing_gold,
        "global": metrics(bi_ranks),
        "late_interaction": (
            {} if args.global_only else {str(k): metrics(li_ranks[k]) for k in ks}
        ),
        "fusion": (
            {}
            if args.global_only
            else {
                str(k): {
                    alpha_key(a): metrics(fusion_ranks[k][alpha_key(a)])
                    for a in alphas
                }
                for k in ks
            }
        ),
        "timing": {
            "offline_code_encode_seconds": code_encode_seconds,
            "batched_query_encode_seconds": query_encode_seconds,
            "global_search_seconds": global_search_seconds,
            "late_interaction_seconds_at_max_k": rerank_seconds,
            "query_encode_ms_per_query": 1000.0 * query_encode_seconds / n_queries,
            "global_search_ms_per_query": 1000.0 * global_search_seconds / n_queries,
            "rerank_ms_per_query_at_max_k": 1000.0 * rerank_seconds / n_queries,
            "online_total_ms_per_query_at_max_k": 1000.0
            * (query_encode_seconds + global_search_seconds + rerank_seconds)
            / n_queries,
        },
        "peak_vram_mib": (
            torch.cuda.max_memory_allocated(device) / (1024.0 ** 2) if device.type == "cuda" else 0.0
        ),
        "peak_reserved_vram_mib": (
            torch.cuda.max_memory_reserved(device) / (1024.0 ** 2) if device.type == "cuda" else 0.0
        ),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    query_ranks_file = None
    if args.save_per_query:
        query_ranks_file = output_dir / "query_ranks.jsonl"
        with query_ranks_file.open("w", encoding="utf-8") as handle:
            for index, (url, global_rank) in enumerate(zip(query_urls, bi_ranks)):
                record = {
                    "query_index": index,
                    "url": url,
                    "global_rank": int(global_rank),
                }
                if not args.global_only:
                    record["late_interaction_rank"] = {
                        str(k): int(li_ranks[k][index]) for k in ks
                    }
                    record["fusion_rank"] = {
                        str(k): {
                            alpha_key(a): int(fusion_ranks[k][alpha_key(a)][index])
                            for a in alphas
                        }
                        for k in ks
                    }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    output_file = output_dir / "grid_results.json"
    output_file.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "saved_to": str(output_file),
        "global_mrr": result["global"]["MRR"],
        "missing_gold": missing_gold,
        "query_ranks": str(query_ranks_file) if query_ranks_file else None,
        "timing": result["timing"],
        "peak_vram_mib": result["peak_vram_mib"],
    }, indent=2))


if __name__ == "__main__":
    main()
