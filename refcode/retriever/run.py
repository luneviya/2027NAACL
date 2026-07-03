# coding=utf-8
"""
Build ReFCode global hard negatives for code search.

For each training query q_i:
  1) encode all train queries with the ReFCode encoder;
  2) retrieve global TopK similar queries by cosine, excluding itself / same url;
  3) sort those TopK candidate query texts by BM25 against q_i;
  4) choose BM25 rank R (default 50 when TopK=500);
  5) save hard_idx[i] = selected query index. Its code is used as q_i's hard negative.

Output is a pickle file with a dict containing key 'hard_idx'.
"""

import argparse
import json
import logging
import math
import os
import pickle
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from tqdm import tqdm
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer

from refcode.retriever.model import Model

logger = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def as_tokens(value):
    if value is None:
        return []
    if isinstance(value, list):
        out = []
        for x in value:
            if isinstance(x, str):
                out.extend(x.split())
            else:
                out.append(str(x))
        return out
    return str(value).split()


def get_nl_tokens(js):
    for key in ["docstring_tokens", "nl_tokens", "query_tokens"]:
        if key in js and js[key]:
            return as_tokens(js[key])
    for key in ["doc", "nl", "query", "docstring"]:
        if key in js and js[key]:
            return str(js[key]).split()
    return []


def get_url(js, idx):
    return js.get("url", js.get("retrieval_idx", str(idx)))


def build_unixcoder_ids(tokens, tokenizer, max_length):
    text = " ".join(tokens)
    toks = tokenizer.tokenize(text)[: max_length - 4]
    toks = [tokenizer.cls_token, "<encoder-only>", tokenizer.sep_token] + toks + [tokenizer.sep_token]
    ids = tokenizer.convert_tokens_to_ids(toks)
    ids += [tokenizer.pad_token_id] * (max_length - len(ids))
    return ids


class QueryDataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return torch.tensor(self.items[idx]["nl_ids"], dtype=torch.long), idx


def load_train_queries(path, tokenizer, nl_length, debug_limit=-1):
    items = []
    with open(path, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            js = json.loads(line)
            nl_tokens = get_nl_tokens(js)
            items.append({
                "idx": idx,
                "nl_tokens": nl_tokens,
                "nl_ids": build_unixcoder_ids(nl_tokens, tokenizer, nl_length),
                "url": get_url(js, idx),
            })
            if debug_limit and debug_limit > 0 and len(items) >= debug_limit:
                break
    return items


@torch.no_grad()
def encode_queries(model, dataset, batch_size, device):
    loader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=batch_size, num_workers=4)
    vecs = []
    model.eval()
    for batch in tqdm(loader, desc="Encoding train queries"):
        input_ids = batch[0].to(device)
        v = model(nl_inputs=input_ids)
        vecs.append(v.detach().cpu())
    return torch.cat(vecs, dim=0).contiguous()


def bm25_choose(anchor_tokens, candidate_indices, all_tokens, bm25_rank=50, k1=1.5, b=0.75):
    """Return selected candidate index by BM25 rank; bm25_rank is 1-based."""
    if not candidate_indices:
        return -1
    docs = [all_tokens[j] for j in candidate_indices]
    n_docs = len(docs)
    avgdl = sum(len(d) for d in docs) / max(1, n_docs)
    if avgdl <= 0:
        avgdl = 1.0

    df = Counter()
    doc_counters = []
    for d in docs:
        c = Counter(d)
        doc_counters.append(c)
        for t in c:
            df[t] += 1

    q_terms = Counter(anchor_tokens)
    scored = []
    for local_pos, (global_idx, d, c) in enumerate(zip(candidate_indices, docs, doc_counters)):
        dl = max(1, len(d))
        score = 0.0
        for term, qtf in q_terms.items():
            if term not in c:
                continue
            # Standard BM25 IDF with smoothing.
            idf = math.log(1.0 + (n_docs - df[term] + 0.5) / (df[term] + 0.5))
            tf = c[term]
            denom = tf + k1 * (1.0 - b + b * dl / avgdl)
            score += idf * (tf * (k1 + 1.0) / denom) * qtf
        # Stable tie break keeps original cosine order.
        scored.append((score, -local_pos, global_idx))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    rank0 = max(0, min(int(bm25_rank) - 1, len(scored) - 1))
    return int(scored[rank0][2])


def build_hard_idx(embeddings, tokens, urls, topk, bm25_rank, chunk_size, device, filter_same_url=True):
    n = embeddings.size(0)
    topk = min(topk, max(1, n - 1))
    emb = embeddings.to(device)
    hard_idx = [-1] * n
    url_to_indices = defaultdict(list)
    if filter_same_url:
        for i, u in enumerate(urls):
            url_to_indices[u].append(i)

    logger.info("Building global Top%d + BM25 rank%d hard negatives for %d queries", topk, bm25_rank, n)
    for start in tqdm(range(0, n, chunk_size), desc="Mining hard negatives"):
        end = min(start + chunk_size, n)
        sims = torch.matmul(emb[start:end], emb.t())
        row = torch.arange(end - start, device=device)
        sims[row, torch.arange(start, end, device=device)] = -1e4
        if filter_same_url:
            for local, i in enumerate(range(start, end)):
                same = url_to_indices.get(urls[i], [])
                if same:
                    sims[local, torch.tensor(same, device=device, dtype=torch.long)] = -1e4
        _, idx = torch.topk(sims, k=topk, dim=1, largest=True, sorted=True)
        idx = idx.cpu().tolist()
        for local, cand in enumerate(idx):
            i = start + local
            cand = [j for j in cand if j != i and (not filter_same_url or urls[j] != urls[i])]
            if not cand:
                # Very rare fallback: first different sample.
                cand = [j for j in range(n) if j != i and (not filter_same_url or urls[j] != urls[i])]
            hard_idx[i] = bm25_choose(tokens[i], cand, tokens, bm25_rank=bm25_rank)
    return hard_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--model_name_or_path", default="DeepSoftwareAnalytics/CoCoSoDa")
    parser.add_argument("--config_name", default="DeepSoftwareAnalytics/CoCoSoDa")
    parser.add_argument("--tokenizer_name", default="DeepSoftwareAnalytics/CoCoSoDa")
    parser.add_argument("--nl_length", type=int, default=128)
    parser.add_argument("--encode_batch_size", type=int, default=128)
    parser.add_argument("--topk", type=int, default=500)
    parser.add_argument("--bm25_rank", type=int, default=50)
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=123456)
    parser.add_argument("--debug_limit", type=int, default=-1)
    parser.add_argument("--no_filter_same_url", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--embeddings_cache", default="", help="optional .pt path to save/load query embeddings")
    args = parser.parse_args()

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO)
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    logger.info("device=%s", device)

    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    config = RobertaConfig.from_pretrained(args.config_name)
    base = RobertaModel.from_pretrained(args.model_name_or_path)
    model = Model(base, hidden_size=getattr(config, "hidden_size", 768))
    model.to(device)

    items = load_train_queries(args.train_data_file, tokenizer, args.nl_length, args.debug_limit)
    logger.info("Loaded %d train queries from %s", len(items), args.train_data_file)
    dataset = QueryDataset(items)
    tokens = [x["nl_tokens"] for x in items]
    urls = [x["url"] for x in items]

    if args.embeddings_cache and os.path.exists(args.embeddings_cache):
        logger.info("Loading embeddings cache from %s", args.embeddings_cache)
        embeddings = torch.load(args.embeddings_cache, map_location="cpu")
    else:
        embeddings = encode_queries(model, dataset, args.encode_batch_size, device)
        if args.embeddings_cache:
            Path(args.embeddings_cache).parent.mkdir(parents=True, exist_ok=True)
            torch.save(embeddings, args.embeddings_cache)
            logger.info("Saved embeddings cache to %s", args.embeddings_cache)

    hard_idx = build_hard_idx(
        embeddings=embeddings,
        tokens=tokens,
        urls=urls,
        topk=args.topk,
        bm25_rank=args.bm25_rank,
        chunk_size=args.chunk_size,
        device=device,
        filter_same_url=not args.no_filter_same_url,
    )
    meta = {
        "hard_idx": hard_idx,
        "train_data_file": args.train_data_file,
        "model_name_or_path": args.model_name_or_path,
        "topk": args.topk,
        "bm25_rank": args.bm25_rank,
        "filter_same_url": not args.no_filter_same_url,
        "num_examples": len(items),
    }
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, "wb") as f:
        pickle.dump(meta, f)
    logger.info("Saved hard_idx to %s", args.output_file)
    logger.info("First 10 hard_idx: %s", hard_idx[:10])


if __name__ == "__main__":
    main()
