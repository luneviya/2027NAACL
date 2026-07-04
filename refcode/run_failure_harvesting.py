# coding=utf-8
"""
Build self-mined retrieval failures for ReFCode code search.

For each training query q_i:
  1) load a trained checkpoint, usually ReFCode+CTRD best checkpoint;
  2) encode all training queries and all training codes;
  3) retrieve TopK codes that the model ranks highly for q_i;
  4) exclude the gold code itself and, optionally, examples with the same URL;
  5) save a list-of-lists self_mined_idx[i] = [j1, j2, ...].

The output is consumed by the refinement runner.
"""

import argparse
import json
import logging
import os
import pickle
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from tqdm import tqdm
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer

from refcode.model import Model

logger = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _as_token_list(value):
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


def get_code_tokens(js):
    """Extract code tokens from CodeSearchNet-style rows."""
    if js.get('function_tokens'):
        return _as_token_list(js.get('function_tokens'))
    if js.get('code_tokens'):
        return _as_token_list(js.get('code_tokens'))
    if js.get('original_string'):
        return str(js.get('original_string')).split()
    if js.get('code'):
        return str(js.get('code')).split()
    return []


def get_nl_tokens(js):
    """Extract query tokens from CodeSearchNet-style rows."""
    if js.get('docstring_tokens'):
        return _as_token_list(js.get('docstring_tokens'))
    if js.get('doc'):
        return str(js.get('doc')).split()
    if js.get('nl'):
        return str(js.get('nl')).split()
    if js.get('query'):
        return str(js.get('query')).split()
    return []


def get_url(js, idx):
    return js.get('url', js.get('retrieval_idx', str(idx)))


def build_unixcoder_ids(tokens, tokenizer, max_length):
    """Convert NL/code tokens to the UniXcoder encoder-only input format."""
    text = ' '.join(tokens) if isinstance(tokens, list) else ' '.join(str(tokens).split())
    toks = tokenizer.tokenize(text)[: max_length - 4]
    toks = [tokenizer.cls_token, '<encoder-only>', tokenizer.sep_token] + toks + [tokenizer.sep_token]
    ids = tokenizer.convert_tokens_to_ids(toks)
    ids += [tokenizer.pad_token_id] * (max_length - len(ids))
    return ids


class IdDataset(Dataset):
    def __init__(self, ids):
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return torch.tensor(self.ids[idx], dtype=torch.long), idx


def load_train_items(path, tokenizer, nl_length, code_length, debug_limit=-1):
    """Load train JSONL rows and tokenize both query and paired code.

    The output order is kept identical to the train file so mined indices can be
    used later as row-aligned failure candidates.
    """
    items = []
    with open(path, encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            js = json.loads(line)
            nl_tokens = get_nl_tokens(js)
            code_tokens = get_code_tokens(js)
            items.append({
                'idx': idx,
                'url': get_url(js, idx),
                'nl_ids': build_unixcoder_ids(nl_tokens, tokenizer, nl_length),
                'code_ids': build_unixcoder_ids(code_tokens, tokenizer, code_length),
            })
            if debug_limit and debug_limit > 0 and len(items) >= debug_limit:
                break
    return items


@torch.no_grad()
def encode_ids(model, ids, batch_size, device, mode='nl', num_workers=4):
    """Encode query or code ids with the current retriever checkpoint."""
    dataset = IdDataset(ids)
    loader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=batch_size, num_workers=num_workers)
    vecs = []
    model.eval()
    for batch in tqdm(loader, desc=f'Encoding {mode}'):
        input_ids = batch[0].to(device)
        if mode == 'nl':
            v = model(nl_inputs=input_ids)
        else:
            v = model(code_inputs=input_ids)
        vecs.append(v.detach().cpu())
    return torch.cat(vecs, dim=0).contiguous()


def resolve_checkpoint(args):
    """Resolve the checkpoint used to harvest model-induced retrieval failures."""
    if args.checkpoint_file:
        return args.checkpoint_file
    if args.output_dir:
        return os.path.join(args.output_dir, 'checkpoint-best-mrr', 'model.bin')
    return ''


def load_model(args, device):
    """Load the base encoder plus an optional trained retriever checkpoint."""
    config = RobertaConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path)
    encoder = RobertaModel.from_pretrained(args.model_name_or_path)
    model = Model(encoder)
    ckpt = resolve_checkpoint(args)
    if ckpt:
        logger.info('Loading checkpoint: %s', ckpt)
        state = torch.load(ckpt, map_location='cpu')
        missing, unexpected = model.load_state_dict(state, strict=False)
        logger.info('Loaded checkpoint with strict=False; missing=%d unexpected=%d', len(missing), len(unexpected))
    model.to(device)
    return model, tokenizer


@torch.no_grad()
def mine(q_vecs, c_vecs, urls, topk, chunk_size=8192, device='cuda', exclude_same_url=True):
    """Mine top-ranked wrong training codes for each training query.

    The gold code and same-URL examples are excluded by default. The saved lists
    approximate retrieval failures produced by the current model.
    """
    n = q_vecs.size(0)
    topk = int(topk)
    url_to_indices = defaultdict(list)
    if exclude_same_url:
        for i, u in enumerate(urls):
            url_to_indices[u].append(i)

    results = []
    q_batch_size = 128
    for start in tqdm(range(0, n, q_batch_size), desc='Mining self hard negatives'):
        end = min(n, start + q_batch_size)
        q = q_vecs[start:end].to(device)
        bsz = q.size(0)
        best_vals = torch.full((bsz, topk), -1e4, device=device)
        best_idx = torch.full((bsz, topk), -1, dtype=torch.long, device=device)

        for c_start in range(0, n, chunk_size):
            c_end = min(n, c_start + chunk_size)
            c = c_vecs[c_start:c_end].to(device)
            sim = torch.matmul(q, c.t())

            # Exclude the gold code itself.
            for local_i, global_i in enumerate(range(start, end)):
                if c_start <= global_i < c_end:
                    sim[local_i, global_i - c_start] = -1e4
                if exclude_same_url:
                    for j in url_to_indices.get(urls[global_i], []):
                        if c_start <= j < c_end:
                            sim[local_i, j - c_start] = -1e4

            local_k = min(topk, c_end - c_start)
            vals, idxs = torch.topk(sim, k=local_k, dim=1)
            idxs = idxs + c_start

            all_vals = torch.cat([best_vals, vals], dim=1)
            all_idxs = torch.cat([best_idx, idxs], dim=1)
            vals2, pos = torch.topk(all_vals, k=topk, dim=1)
            idxs2 = torch.gather(all_idxs, 1, pos)
            best_vals, best_idx = vals2, idxs2

        for row in best_idx.cpu().tolist():
            # Remove any fallback -1 and duplicate just in case.
            clean = []
            seen = set()
            for x in row:
                x = int(x)
                if x >= 0 and x not in seen:
                    clean.append(x)
                    seen.add(x)
            if not clean:
                clean = [int((len(results) + 1) % n)]
            results.append(clean)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_file', required=True, type=str)
    parser.add_argument('--output_file', required=True, type=str)
    parser.add_argument('--output_dir', default='', type=str, help='directory containing checkpoint-best-mrr/model.bin')
    parser.add_argument('--checkpoint_file', default='', type=str, help='explicit model.bin checkpoint path')
    parser.add_argument('--model_name_or_path', default='DeepSoftwareAnalytics/CoCoSoDa', type=str)
    parser.add_argument('--config_name', default='DeepSoftwareAnalytics/CoCoSoDa', type=str)
    parser.add_argument('--tokenizer_name', default='DeepSoftwareAnalytics/CoCoSoDa', type=str)
    parser.add_argument('--nl_length', default=128, type=int)
    parser.add_argument('--code_length', default=256, type=int)
    parser.add_argument('--encode_batch_size', default=128, type=int)
    parser.add_argument('--topk', default=32, type=int, help='number of mined wrong codes to save per query')
    parser.add_argument('--chunk_size', default=8192, type=int, help='code-vector chunk size for retrieval')
    parser.add_argument('--seed', default=123456, type=int)
    parser.add_argument('--debug_limit', default=-1, type=int)
    parser.add_argument('--include_same_url', action='store_true', help='do not exclude examples with the same url')
    parser.add_argument('--num_workers', default=4, type=int)
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('device=%s', device)

    model, tokenizer = load_model(args, device)
    logger.info('Loading train items: %s', args.train_data_file)
    items = load_train_items(args.train_data_file, tokenizer, args.nl_length, args.code_length, args.debug_limit)
    logger.info('Loaded %d train items', len(items))

    nl_ids = [x['nl_ids'] for x in items]
    code_ids = [x['code_ids'] for x in items]
    urls = [x['url'] for x in items]

    q_vecs = encode_ids(model, nl_ids, args.encode_batch_size, device, mode='nl', num_workers=args.num_workers)
    c_vecs = encode_ids(model, code_ids, args.encode_batch_size, device, mode='code', num_workers=args.num_workers)

    mined_idx = mine(
        q_vecs=q_vecs,
        c_vecs=c_vecs,
        urls=urls,
        topk=args.topk,
        chunk_size=args.chunk_size,
        device=device,
        exclude_same_url=not args.include_same_url,
    )

    out = {
        'self_mined_idx': mined_idx,
        'topk': args.topk,
        'train_data_file': args.train_data_file,
        'checkpoint': resolve_checkpoint(args),
        'exclude_same_url': not args.include_same_url,
    }
    # Output format consumed by refinement: pickle(dict), with self_mined_idx[i]
    # containing candidate row indices for train example i.
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, 'wb') as f:
        pickle.dump(out, f)
    logger.info('Saved self-mined hard negatives to %s', args.output_file)
    logger.info('First example mined indices: %s', mined_idx[0][:min(10, len(mined_idx[0]))] if mined_idx else [])


if __name__ == '__main__':
    main()
