# ReFCode inference-only token-level late-interaction reranker for code search.
# ReFCode inference-only token-level late-interaction reranker for code search.
# It loads your best bi-encoder checkpoint, retrieves topK candidates,
# then reranks topK using ColBERT-style MaxSim token-level interaction.

import argparse
import json
import os
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from tqdm import tqdm
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer

from refcode.refinement.model import Model


def _as_token_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        out = []
        for item in value:
            if isinstance(item, str):
                out.extend(item.split())
            else:
                out.append(str(item))
        return out
    if isinstance(value, str):
        return value.split()
    return str(value).split()


def _extract_code_tokens(js: Dict[str, Any]) -> List[str]:
    for key in ["function_tokens", "code_tokens"]:
        if key in js and js[key]:
            return _as_token_list(js[key])
    for key in ["original_string", "code"]:
        if key in js and js[key]:
            return str(js[key]).split()
    return []


def _extract_nl_tokens(js: Dict[str, Any]) -> List[str]:
    if "docstring_tokens" in js and js["docstring_tokens"]:
        return _as_token_list(js["docstring_tokens"])
    for key in ["doc", "nl"]:
        if key in js and js[key]:
            return str(js[key]).split()
    return []


def _build_unixcoder_ids(tokens, tokenizer, max_length: int):
    text = " ".join(tokens) if isinstance(tokens, list) else " ".join(str(tokens).split())
    toks = tokenizer.tokenize(text)[: max_length - 4]
    toks = [tokenizer.cls_token, "<encoder-only>", tokenizer.sep_token] + toks + [tokenizer.sep_token]
    ids = tokenizer.convert_tokens_to_ids(toks)
    ids += [tokenizer.pad_token_id] * (max_length - len(ids))
    return ids[:max_length]


def read_jsonl(path: str):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


class CodeSearchDataset(Dataset):
    def __init__(self, file_path: str, tokenizer, code_length: int, nl_length: int, mode: str):
        self.examples = []
        data = read_jsonl(file_path)
        for js in data:
            url = js.get("url", js.get("retrieval_idx", ""))
            code_ids = _build_unixcoder_ids(_extract_code_tokens(js), tokenizer, code_length)
            nl_ids = _build_unixcoder_ids(_extract_nl_tokens(js), tokenizer, nl_length)
            self.examples.append({"url": url, "code_ids": code_ids, "nl_ids": nl_ids})
        self.mode = mode

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        if self.mode == "query":
            return torch.tensor(ex["nl_ids"], dtype=torch.long), idx
        return torch.tensor(ex["code_ids"], dtype=torch.long), idx


def collate_ids(batch):
    ids = torch.stack([x[0] for x in batch], dim=0)
    idx = torch.tensor([x[1] for x in batch], dtype=torch.long)
    return ids, idx


def load_model(args):
    RobertaConfig.from_pretrained(args.config_name or args.model_name_or_path)
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name or args.model_name_or_path)
    encoder = RobertaModel.from_pretrained(args.model_name_or_path)
    model = Model(encoder)

    state = torch.load(args.loaded_model_filename, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[Load] checkpoint={args.loaded_model_filename}")
    print(f"[Load] missing={len(missing)}, unexpected={len(unexpected)}")

    model.to(args.device)
    model.eval()
    return model, tokenizer


def encode_pooled(model, dataloader, device, is_code: bool, fp16: bool = True):
    vecs, indices = [], []
    desc = "Encoding pooled code" if is_code else "Encoding pooled query"
    model.eval()
    with torch.no_grad():
        for ids, idx in tqdm(dataloader, desc=desc):
            ids = ids.to(device)
            with torch.cuda.amp.autocast(enabled=fp16 and device.type == "cuda"):
                out = model(code_inputs=ids) if is_code else model(nl_inputs=ids)
            vecs.append(out.detach().float().cpu())
            indices.append(idx)
    vecs = torch.cat(vecs, dim=0)
    indices = torch.cat(indices, dim=0)
    order = torch.argsort(indices)
    return vecs[order].contiguous()


def get_last_hidden(model, input_ids):
    base = model.module if hasattr(model, "module") else model
    attention_mask = input_ids.ne(1)
    try:
        outputs = base.encoder(input_ids, attention_mask=attention_mask, return_dict=True)
        return outputs.last_hidden_state
    except TypeError:
        outputs = base.encoder(input_ids, attention_mask=attention_mask)
        return outputs[0]


def token_mask(input_ids, tokenizer):
    mask = input_ids.ne(tokenizer.pad_token_id)
    special_ids = [
        tokenizer.cls_token_id,
        tokenizer.sep_token_id,
        tokenizer.pad_token_id,
        tokenizer.convert_tokens_to_ids("<encoder-only>"),
    ]
    for sid in special_ids:
        if sid is not None and sid >= 0:
            mask = mask & input_ids.ne(sid)
    return mask


def late_interaction_score(q_tok, q_mask, c_tok, c_mask):
    q_tok = F.normalize(q_tok, p=2, dim=-1)
    c_tok = F.normalize(c_tok, p=2, dim=-1)
    sim = torch.bmm(q_tok, c_tok.transpose(1, 2))
    sim = sim.masked_fill(~c_mask.unsqueeze(1), -1e4)
    max_sim = sim.max(dim=2).values
    q_mask_f = q_mask.float()
    return (max_sim * q_mask_f).sum(dim=1) / q_mask_f.sum(dim=1).clamp_min(1.0)


def zscore(x):
    x = np.asarray(x, dtype=np.float32)
    return (x - x.mean()) / (x.std() + 1e-6)


def mrr_from_ranks(ranks):
    return float(np.mean([1.0 / r if r > 0 else 0.0 for r in ranks]))


def recall_from_ranks(ranks, k):
    return float(np.mean([1.0 if r > 0 and r <= k else 0.0 for r in ranks]))


def rerank(args, model, tokenizer, query_dataset, code_dataset, query_vecs, code_vecs):
    device = args.device
    code_urls = [ex["url"] for ex in code_dataset.examples]
    query_urls = [ex["url"] for ex in query_dataset.examples]
    url_to_code_idx = {}
    for i, u in enumerate(code_urls):
        if u not in url_to_code_idx:
            url_to_code_idx[u] = i

    top_k = int(args.top_k)
    alpha = float(args.fusion_alpha)

    bi_ranks, li_ranks, fusion_ranks = [], [], []
    code_vecs_t = code_vecs.to(device)
    query_vecs_t = query_vecs.to(device)

    for qi in tqdm(range(len(query_dataset)), desc=f"Late interaction rerank top{top_k}"):
        qv = query_vecs_t[qi: qi + 1]
        scores = torch.matmul(qv, code_vecs_t.t()).squeeze(0)

        gold_idx = url_to_code_idx.get(query_urls[qi], None)
        if gold_idx is None:
            bi_ranks.append(0)
            li_ranks.append(0)
            fusion_ranks.append(0)
            continue

        gold_score = scores[gold_idx]
        bi_rank = int((scores > gold_score).sum().item()) + 1
        bi_ranks.append(bi_rank)

        k = min(top_k, scores.size(0))
        top_scores, top_idx = torch.topk(scores, k=k, dim=0)
        top_idx_np = top_idx.detach().cpu().numpy()
        bi_scores_np = top_scores.detach().float().cpu().numpy()

        if gold_idx not in set(top_idx_np.tolist()):
            li_ranks.append(bi_rank)
            fusion_ranks.append(bi_rank)
            continue

        q_ids = torch.tensor(query_dataset.examples[qi]["nl_ids"], dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=args.fp16 and device.type == "cuda"):
                q_hidden = get_last_hidden(model, q_ids)
            q_mask = token_mask(q_ids, tokenizer)
            q_hidden = q_hidden.repeat(k, 1, 1)
            q_mask = q_mask.repeat(k, 1)

        li_scores = []
        for start in range(0, k, args.rerank_batch_size):
            end = min(start + args.rerank_batch_size, k)
            cand_ids = [code_dataset.examples[int(cid)]["code_ids"] for cid in top_idx_np[start:end]]
            c_ids = torch.tensor(cand_ids, dtype=torch.long, device=device)
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=args.fp16 and device.type == "cuda"):
                    c_hidden = get_last_hidden(model, c_ids)
                c_mask = token_mask(c_ids, tokenizer)
                li = late_interaction_score(
                    q_hidden[start:end].float(),
                    q_mask[start:end],
                    c_hidden.float(),
                    c_mask,
                )
                li_scores.append(li.detach().cpu())
        li_scores_np = torch.cat(li_scores, dim=0).numpy()

        li_order = np.argsort(li_scores_np)[::-1]
        li_ids = top_idx_np[li_order]
        li_pos = int(np.where(li_ids == gold_idx)[0][0]) + 1
        li_ranks.append(li_pos)

        final_scores = alpha * zscore(bi_scores_np) + (1.0 - alpha) * zscore(li_scores_np)
        fusion_order = np.argsort(final_scores)[::-1]
        fusion_ids = top_idx_np[fusion_order]
        fusion_pos = int(np.where(fusion_ids == gold_idx)[0][0]) + 1
        fusion_ranks.append(fusion_pos)

    result = {
        "bi_mrr": mrr_from_ranks(bi_ranks),
        "late_interaction_only_mrr": mrr_from_ranks(li_ranks),
        "fusion_mrr": mrr_from_ranks(fusion_ranks),

        "bi_recall@1": recall_from_ranks(bi_ranks, 1),
        "bi_recall@5": recall_from_ranks(bi_ranks, 5),
        "bi_recall@10": recall_from_ranks(bi_ranks, 10),
        "bi_recall@50": recall_from_ranks(bi_ranks, 50),

        "late_interaction_only_recall@1": recall_from_ranks(li_ranks, 1),
        "late_interaction_only_recall@5": recall_from_ranks(li_ranks, 5),
        "late_interaction_only_recall@10": recall_from_ranks(li_ranks, 10),
        "late_interaction_only_recall@50": recall_from_ranks(li_ranks, 50),

        "fusion_recall@1": recall_from_ranks(fusion_ranks, 1),
        "fusion_recall@5": recall_from_ranks(fusion_ranks, 5),
        "fusion_recall@10": recall_from_ranks(fusion_ranks, 10),
        "fusion_recall@50": recall_from_ranks(fusion_ranks, 50),

        "top_k": top_k,
        "fusion_alpha": alpha,
        "num_queries": len(query_dataset),
    }

    os.makedirs(args.output_dir, exist_ok=True)
    out_file = os.path.join(args.output_dir, f"late_interaction_{args.lang}_top{top_k}_alpha{alpha}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("\n===== Late Interaction Result =====")
    for k, v in result.items():
        print(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}")
    print(f"saved_to: {out_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, default="javascript")
    parser.add_argument("--model_name_or_path", type=str, default="DeepSoftwareAnalytics/CoCoSoDa")
    parser.add_argument("--config_name", type=str, default="DeepSoftwareAnalytics/CoCoSoDa")
    parser.add_argument("--tokenizer_name", type=str, default="DeepSoftwareAnalytics/CoCoSoDa")
    parser.add_argument("--loaded_model_filename", type=str, required=True)
    parser.add_argument("--eval_data_file", type=str, default="dataset/javascript/test.jsonl")
    parser.add_argument("--codebase_file", type=str, default="dataset/javascript/codebase.jsonl")
    parser.add_argument("--output_dir", type=str, default="./saved_models/refcode/rerank")
    parser.add_argument("--code_length", type=int, default=256)
    parser.add_argument("--nl_length", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--rerank_batch_size", type=int, default=64)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--fusion_alpha", type=float, default=0.5)
    parser.add_argument("--fp16", type=int, default=1)
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.fp16 = bool(args.fp16)

    print("===== Late Interaction Rerank Args =====")
    for k, v in sorted(vars(args).items()):
        print(f"{k}: {v}")

    model, tokenizer = load_model(args)

    query_dataset = CodeSearchDataset(args.eval_data_file, tokenizer, args.code_length, args.nl_length, "query")
    code_dataset = CodeSearchDataset(args.codebase_file, tokenizer, args.code_length, args.nl_length, "code")

    query_loader = DataLoader(query_dataset, sampler=SequentialSampler(query_dataset),
                              batch_size=args.eval_batch_size, collate_fn=collate_ids, num_workers=4)
    code_loader = DataLoader(code_dataset, sampler=SequentialSampler(code_dataset),
                             batch_size=args.eval_batch_size, collate_fn=collate_ids, num_workers=4)

    query_vecs = encode_pooled(model, query_loader, args.device, is_code=False, fp16=args.fp16)
    code_vecs = encode_pooled(model, code_loader, args.device, is_code=True, fp16=args.fp16)

    rerank(args, model, tokenizer, query_dataset, code_dataset, query_vecs, code_vecs)


if __name__ == "__main__":
    main()
