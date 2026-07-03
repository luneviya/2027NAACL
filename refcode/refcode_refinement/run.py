# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from unittest import removeResult
import torch.nn.functional as F
import argparse
import logging
import os
import pickle
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import json
import math
import re
from collections import Counter
from random import choice
import numpy as np
from itertools import cycle
from refcode.refcode_refinement.model import Model
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                  RobertaConfig, RobertaModel, RobertaTokenizer)

logger = logging.getLogger(__name__)
from tqdm import tqdm
import multiprocessing
cpu_cont = 16

from refcode.utils.parser import DFG_python,DFG_java,DFG_ruby,DFG_go,DFG_php,DFG_javascript
from refcode.utils.parser import (remove_comments_and_docstrings,
                   tree_to_token_index,
                   index_to_code_token,
                   tree_to_variable_index)
from tree_sitter import Language, Parser
sys.path.append("dataset")
from refcode.utils.utils import save_json_data, save_pickle_data
dfg_function={
    'python':DFG_python,
    'java':DFG_java,
    'ruby':DFG_ruby,
    'go':DFG_go,
    'php':DFG_php,
    'javascript':DFG_javascript
}

parsers={}        
for lang in dfg_function:
    language_library = os.path.join(os.path.dirname(__file__), '..', 'utils', 'parser', 'my-languages.so')
    LANGUAGE = Language(language_library, lang)
    parser = Parser()
    parser.set_language(LANGUAGE) 
    parser = [parser,dfg_function[lang]]    
    parsers[lang]= parser
    

ruby_special_token = ['keyword', 'identifier', 'separators', 'simple_symbol', 'constant', 'instance_variable',
 'operator', 'string_content', 'integer', 'escape_sequence', 'comment', 'hash_key_symbol',
  'global_variable', 'heredoc_beginning', 'heredoc_content', 'heredoc_end', 'class_variable',]

java_special_token = ['keyword', 'identifier', 'type_identifier',  'separators', 'operator', 'decimal_integer_literal',
 'void_type', 'string_literal', 'decimal_floating_point_literal', 
 'boolean_type', 'null_literal', 'comment', 'hex_integer_literal', 'character_literal']

go_special_token = ['keyword', 'identifier', 'separators', 'type_identifier', 'int_literal', 'operator', 
'field_identifier', 'package_identifier', 'comment',  'escape_sequence', 'raw_string_literal',
'rune_literal', 'label_name', 'float_literal']

javascript_special_token =['keyword', 'separators', 'identifier', 'property_identifier', 'operator', 
'number', 'string_fragment', 'comment', 'regex_pattern', 'shorthand_property_identifier_pattern', 
'shorthand_property_identifier', 'regex_flags', 'escape_sequence', 'statement_identifier']

php_special_token =['text', 'php_tag', 'name', 'operator', 'keyword', 'string', 'integer', 'separators', 'comment', 
'escape_sequence', 'ERROR',  'boolean', 'namespace', 'class', 'extends']

python_special_token =['keyword', 'identifier', 'separators', 'operator', '"', 'integer', 
'comment', 'none', 'escape_sequence']


special_token={
    'python':python_special_token,
    'java':java_special_token,
    'ruby':ruby_special_token,
    'go':go_special_token,
    'php':php_special_token,
    'javascript':javascript_special_token
}

all_special_token = []
for key, value in special_token.items():
    all_special_token = list(set(all_special_token ).union(set(value)))

def lalign(x, y, alpha=2):
    x = torch.tensor(x)
    y= torch.tensor(y)
    return (x - y).norm(dim=1).pow(alpha).mean()
    # code2nl_pos = torch.einsum('nc,nc->n', [x, y]).unsqueeze(-1)

    # return code2nl_pos.mean()

def lunif(x, t=2):
    x = torch.tensor(x)
    sq_pdist = torch.pdist(x, p=2).pow(2)
    return sq_pdist.mul(-t).exp().mean().log()



def cal_r1_r5_r10(ranks):
    r1,r5,r10= 0,0,0
    data_len= len(ranks)
    for item in ranks:
        if item >=1:
            r1 +=1
            r5 += 1 
            r10 += 1
        elif item >=0.2:
            r5+= 1
            r10+=1
        elif item >=0.1:
            r10 +=1
    result = {"R@1":round(r1/data_len,3), "R@5": round(r5/data_len,3),  "R@10": round(r10/data_len,3)}
    return result

#remove comments, tokenize code and extract dataflow                                        
def extract_dataflow(code, parser,lang):
    #remove comments
    try:
        code=remove_comments_and_docstrings(code,lang)
    except:
        pass    
    #obtain dataflow
    if lang=="php":
        code="<?php"+code+"?>"    
    try:
        tree = parser[0].parse(bytes(code,'utf8'))    
        root_node = tree.root_node  
        tokens_index=tree_to_token_index(root_node)     
        code=code.split('\n')
        code_tokens=[index_to_code_token(x,code) for x in tokens_index]  
        index_to_code={}
        for idx,(index,code) in enumerate(zip(tokens_index,code_tokens)):
            index_to_code[index]=(idx,code)  
        try:
            DFG,_=parser[1](root_node,index_to_code,{}) 
        except:
            DFG=[]
        DFG=sorted(DFG,key=lambda x:x[1])
        indexs=set()
        for d in DFG:
            if len(d[-1])!=0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        new_DFG=[]
        for d in DFG:
            if d[1] in indexs:
                new_DFG.append(d)
        dfg=new_DFG
    except:
        dfg=[]
    return code_tokens,dfg

#remove comments, tokenize code and extract dataflow                                        
def tokenizer_source_code(code, parser,lang):
    #remove comments
    try:
        code=remove_comments_and_docstrings(code,lang)
    except:
        pass    
    #obtain dataflow
    if lang=="php":
        code="<?php"+code+"?>"    
    try:
        tree = parser[0].parse(bytes(code,'utf8'))    
        root_node = tree.root_node  
        tokens_index=tree_to_token_index(root_node)     
        code=code.split('\n')
        code_tokens=[index_to_code_token(x,code) for x in tokens_index]  
    except:
        dfg=[]
    return code_tokens

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 code_tokens,
                 code_ids,
                #  position_idx,
                #  dfg_to_code,
                #  dfg_to_dfg,                 
                 nl_tokens,
                 nl_ids,
                 url,

    ):
        self.code_tokens = code_tokens
        self.code_ids = code_ids
        # self.position_idx=position_idx
        # self.dfg_to_code=dfg_to_code
        # self.dfg_to_dfg=dfg_to_dfg        
        self.nl_tokens = nl_tokens
        self.nl_ids = nl_ids
        self.url=url


class TypeAugInputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 code_tokens,
                 code_ids,
                #  position_idx,
                 code_type,
                 code_type_ids,                 
                 nl_tokens,
                 nl_ids,
                 url,

    ):
        self.code_tokens = code_tokens
        self.code_ids = code_ids
        # self.position_idx=position_idx
        self.code_type=code_type
        self.code_type_ids=code_type_ids        
        self.nl_tokens = nl_tokens
        self.nl_ids = nl_ids
        self.url=url

def convert_examples_to_features(js):
    js,tokenizer,args=js
    #code
    if args.lang == "java_mini":
        parser=parsers["java"]
    else:
        parser=parsers[js["language"]]
    # code
    code_tokens=tokenizer_source_code(js['original_string'],parser,args.lang)
    code_tokens=" ".join(code_tokens[:args.code_length-2])
    code_tokens=tokenizer.tokenize(code_tokens)[:args.code_length-2]
    code_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    code_ids =  tokenizer.convert_tokens_to_ids(code_tokens)
    padding_length = args.code_length - len(code_ids)
    code_ids+=[tokenizer.pad_token_id]*padding_length   

    #nl
    nl=' '.join(js['docstring_tokens'])
    nl_tokens=tokenizer.tokenize(nl)[:args.nl_length-2]
    nl_tokens =[tokenizer.cls_token]+nl_tokens+[tokenizer.sep_token]
    nl_ids =  tokenizer.convert_tokens_to_ids(nl_tokens)
    padding_length = args.nl_length - len(nl_ids)
    nl_ids+=[tokenizer.pad_token_id]*padding_length  

    return InputFeatures(code_tokens,code_ids,nl_tokens,nl_ids,js['url'])


def convert_examples_to_features_aug_type(js):
    js,tokenizer,args=js
    #code
    if args.lang == "java_mini":
        parser=parsers["java"]
    else:
        parser=parsers[js["language"]]
    # code
    token_type_role = js[ 'bpe_token_type_role']
    code_token = [item[0] for item in token_type_role]
    # code = ' '.join(code_token[:args.code_length-4])
    # code_tokens = tokenizer.tokenize(code)[:args.code_length-4]
    code_tokens = code_token[:args.code_length-4]
    code_tokens =[tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+code_tokens+[tokenizer.sep_token]
    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    padding_length = args.code_length - len(code_ids)
    code_ids += [tokenizer.pad_token_id]*padding_length

    # code type
    code_type_token = [item[-1] for item in token_type_role]
    # code_type= ' '.join(code_type_token[:args.code_length-4])
    # code_type_tokens = tokenizer.tokenize(code_type)[:args.code_length-4]
    code_type_tokens = code_type_token[:args.code_length-4]
    code_type_tokens =[tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+code_type_tokens+[tokenizer.sep_token]
    code_type_ids = tokenizer.convert_tokens_to_ids(code_type_tokens)
    padding_length = args.code_length - len(code_type_ids)
    code_type_ids += [tokenizer.pad_token_id]*padding_length

    #nl
    nl=' '.join(js['docstring_tokens'])
    nl_tokens = tokenizer.tokenize(nl)[:args.nl_length-4]
    nl_tokens = [tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+nl_tokens+[tokenizer.sep_token]
    nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)
    padding_length = args.nl_length - len(nl_ids)
    nl_ids += [tokenizer.pad_token_id]*padding_length 

    return TypeAugInputFeatures(code_tokens,code_ids,code_type_tokens,code_type_ids,nl_tokens,nl_ids,js['url'])
  

class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None,pool=None):
        self.args=args
        prefix=file_path.split('/')[-1][:-6]
        cache_file=args.output_dir+'/'+prefix+'.pkl'
        n_debug_samples = args.n_debug_samples
        # if 'codebase' in file_path:
        #     n_debug_samples = 100000
        if 'train' in file_path:
            self.split = "train"
        else:
            self.split = "other"
        if os.path.exists(cache_file):
            self.examples=pickle.load(open(cache_file,'rb'))
            if args.debug:
                self.examples= self.examples[:n_debug_samples]
        else:
            self.examples = []
            data=[]
            if args.debug:
                with open(file_path, encoding="utf-8") as f:
                    for line in f:
                        line=line.strip()
                        js=json.loads(line)
                        data.append((js,tokenizer,args))
                        if len(data) >= n_debug_samples:
                            break
            else:
                with open(file_path, encoding="utf-8") as f:
                    for line in f:
                        line=line.strip()
                        js=json.loads(line)
                        data.append((js,tokenizer,args))
            
            if self.args.data_aug_type == "replace_type":
                self.examples=pool.map(convert_examples_to_features_aug_type, tqdm(data,total=len(data)))
            else:
                self.examples=pool.map(convert_examples_to_features, tqdm(data,total=len(data)))
            
        if 'train' in file_path:
            for idx, example in enumerate(self.examples[:3]):
                logger.info("*** Example ***")
                logger.info("idx: {}".format(idx))
                logger.info("code_tokens: {}".format([x.replace('\u0120','_') for x in example.code_tokens]))
                logger.info("code_ids: {}".format(' '.join(map(str, example.code_ids))))             
                logger.info("nl_tokens: {}".format([x.replace('\u0120','_') for x in example.nl_tokens]))
                logger.info("nl_ids: {}".format(' '.join(map(str, example.nl_ids))))          
                
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item): 
        if self.args.data_aug_type == "replace_type":
            return (torch.tensor(self.examples[item].code_ids),
                    torch.tensor(self.examples[item].code_type_ids),
                    torch.tensor(self.examples[item].nl_ids))
        else:
            return (torch.tensor(self.examples[item].code_ids),
                    torch.tensor(self.examples[item].nl_ids))

        

class ReFCodeInputFeatures(object):
    """Features for ReFCode uncertainty-aware training with optional LLM augmentations."""
    def __init__(self, code_tokens, code_ids, nl_tokens, nl_ids, url,
                 aug_nl_tokens=None, aug_nl_ids=None, has_aug_nl=0,
                 aug_code_tokens=None, aug_code_ids=None, has_aug_code=0,
                 raw_nl_tokens=None, raw_code_tokens=None):
        self.code_tokens = code_tokens
        self.code_ids = code_ids
        self.nl_tokens = nl_tokens
        self.nl_ids = nl_ids
        self.url = url
        self.aug_nl_tokens = aug_nl_tokens if aug_nl_tokens is not None else nl_tokens
        self.aug_nl_ids = aug_nl_ids if aug_nl_ids is not None else nl_ids
        self.has_aug_nl = int(has_aug_nl)
        self.aug_code_tokens = aug_code_tokens if aug_code_tokens is not None else code_tokens
        self.aug_code_ids = aug_code_ids if aug_code_ids is not None else code_ids
        self.has_aug_code = int(has_aug_code)
        self.raw_nl_tokens = raw_nl_tokens if raw_nl_tokens is not None else nl_tokens
        self.raw_code_tokens = raw_code_tokens if raw_code_tokens is not None else code_tokens


def _as_token_list(value):
    if value is None:
        return None
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


def _first_existing(js, keys):
    for key in keys:
        if key in js and js[key] not in [None, "", []]:
            return js[key]
    return None


def _extract_original_code_tokens(js):
    if 'function_tokens' in js and js['function_tokens']:
        return _as_token_list(js['function_tokens'])
    if 'code_tokens' in js and js['code_tokens']:
        return _as_token_list(js['code_tokens'])
    if 'original_string' in js and js['original_string']:
        return str(js['original_string']).split()
    if 'code' in js and js['code']:
        return str(js['code']).split()
    return []


def _extract_original_nl_tokens(js):
    if 'docstring_tokens' in js and js['docstring_tokens']:
        return _as_token_list(js['docstring_tokens'])
    if 'doc' in js and js['doc']:
        return str(js['doc']).split()
    if 'nl' in js and js['nl']:
        return str(js['nl']).split()
    return []


def _extract_aug_nl_tokens(js):
    value = _first_existing(js, [
        'aug_docstring_tokens', 'aug_nl_tokens', 'nl_aug_tokens', 'llm_docstring_tokens',
        'llm_nl_tokens', 'generated_docstring_tokens', 'generated_nl_tokens',
        'semantic_nl_tokens', 'rewritten_docstring_tokens',
        'aug_docstring', 'docstring_aug', 'aug_nl', 'nl_aug', 'llm_docstring',
        'llm_nl', 'generated_docstring', 'generated_nl', 'semantic_nl', 'rewritten_docstring',
    ])
    return _as_token_list(value)


def _extract_aug_code_tokens(js):
    value = _first_existing(js, [
        'aug_code_tokens', 'code_aug_tokens', 'llm_code_tokens', 'generated_code_tokens',
        'semantic_code_tokens', 'aug_function_tokens', 'function_tokens_aug',
        'aug_code', 'code_aug', 'llm_code', 'generated_code', 'semantic_code',
        'aug_original_string', 'original_string_aug',
    ])
    return _as_token_list(value)


def _build_unixcoder_ids(token_source, tokenizer, max_length):
    token_source = token_source if token_source is not None else []
    text = ' '.join(token_source) if isinstance(token_source, list) else ' '.join(str(token_source).split())
    tokens = tokenizer.tokenize(text)[:max_length-4]
    tokens = [tokenizer.cls_token, "<encoder-only>", tokenizer.sep_token] + tokens + [tokenizer.sep_token]
    ids = tokenizer.convert_tokens_to_ids(tokens)
    ids += [tokenizer.pad_token_id] * (max_length - len(ids))
    return tokens, ids


def convert_examples_to_features_unixcoder(js,tokenizer,args):
    """convert examples to token ids; also keeps optional LLM-generated NL/code views."""
    code_raw_tokens = _extract_original_code_tokens(js)
    nl_raw_tokens = _extract_original_nl_tokens(js)

    code_tokens, code_ids = _build_unixcoder_ids(code_raw_tokens, tokenizer, args.code_length)
    nl_tokens, nl_ids = _build_unixcoder_ids(nl_raw_tokens, tokenizer, args.nl_length)

    aug_nl_raw_tokens = _extract_aug_nl_tokens(js)
    has_aug_nl = 1 if aug_nl_raw_tokens else 0
    if has_aug_nl:
        aug_nl_tokens, aug_nl_ids = _build_unixcoder_ids(aug_nl_raw_tokens, tokenizer, args.nl_length)
    else:
        aug_nl_tokens, aug_nl_ids = nl_tokens, nl_ids

    aug_code_raw_tokens = _extract_aug_code_tokens(js)
    has_aug_code = 1 if aug_code_raw_tokens else 0
    if has_aug_code:
        aug_code_tokens, aug_code_ids = _build_unixcoder_ids(aug_code_raw_tokens, tokenizer, args.code_length)
    else:
        aug_code_tokens, aug_code_ids = code_tokens, code_ids

    url = js['url'] if 'url' in js else js.get('retrieval_idx', '')
    return ReFCodeInputFeatures(
        code_tokens, code_ids, nl_tokens, nl_ids, url,
        aug_nl_tokens=aug_nl_tokens, aug_nl_ids=aug_nl_ids, has_aug_nl=has_aug_nl,
        aug_code_tokens=aug_code_tokens, aug_code_ids=aug_code_ids, has_aug_code=has_aug_code,
        raw_nl_tokens=nl_raw_tokens, raw_code_tokens=code_raw_tokens,
    )

class TextDataset_unixcoder(Dataset):
    def __init__(self, tokenizer, args, file_path=None, pooler=None):
        self.args = args
        self.file_path = file_path
        self.split = "train" if file_path and "train" in os.path.basename(file_path) else "other"
        self.examples = []
        data = []
        n_debug_samples = args.n_debug_samples
        with open(file_path, encoding="utf-8") as f:
            if "jsonl" in file_path:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    js = json.loads(line)
                    if 'function_tokens' in js and 'code_tokens' not in js:
                        js['code_tokens'] = js['function_tokens']
                    data.append(js)
                    if args.debug and len(data) >= n_debug_samples:
                        break
            elif "codebase" in file_path or "code_idx_map" in file_path:
                js = json.load(f)
                for key in js:
                    temp = {}
                    temp['code_tokens'] = key.split()
                    temp["retrieval_idx"] = js[key]
                    temp['doc'] = ""
                    temp['docstring_tokens'] = ""
                    data.append(temp)
                    if args.debug and len(data) >= n_debug_samples:
                        break
            elif "json" in file_path:
                for js in json.load(f):
                    data.append(js)
                    if args.debug and len(data) >= n_debug_samples:
                        break
        for js in data:
            self.examples.append(convert_examples_to_features_unixcoder(js,tokenizer,args))

        self.hard_idx = None
        if self.split == "train" and getattr(args, 'refcode_hard_idx_file', ''):
            hard_path = args.refcode_hard_idx_file
            logger.info("Loading ReFCode global hard negative index from %s", hard_path)
            with open(hard_path, 'rb') as f:
                hard_obj = pickle.load(f)
            if isinstance(hard_obj, dict):
                hard_idx = hard_obj.get('hard_idx') or hard_obj.get('hard_indices')
            else:
                hard_idx = hard_obj
            if hard_idx is None:
                raise ValueError("Hard negative file must contain a list or a dict with key 'hard_idx'.")
            hard_idx = [int(x) for x in hard_idx]
            if len(hard_idx) < len(self.examples):
                raise ValueError(f"Hard negative index length {len(hard_idx)} is smaller than dataset length {len(self.examples)}.")
            self.hard_idx = hard_idx
            logger.info("Loaded ReFCode global hard negatives: %d entries", len(self.hard_idx))

        self.self_mined_idx = None
        if self.split == "train" and getattr(args, 'self_mined_idx_file', ''):
            mined_path = args.self_mined_idx_file
            logger.info("Loading self-mined hard negatives from %s", mined_path)
            with open(mined_path, 'rb') as f:
                mined_obj = pickle.load(f)
            if isinstance(mined_obj, dict):
                mined_idx = (mined_obj.get('self_mined_idx') or mined_obj.get('mined_idx')
                             or mined_obj.get('hard_idx') or mined_obj.get('hard_indices'))
            else:
                mined_idx = mined_obj
            if mined_idx is None:
                raise ValueError("Self-mined file must contain a list/list-of-lists or a dict with key 'self_mined_idx'.")
            if len(mined_idx) < len(self.examples):
                raise ValueError(f"Self-mined index length {len(mined_idx)} is smaller than dataset length {len(self.examples)}.")
            topk_keep = int(getattr(args, 'self_mined_topk', 0))
            cleaned = []
            for row in mined_idx[:len(self.examples)]:
                if isinstance(row, (list, tuple, np.ndarray)):
                    vals = [int(x) for x in row]
                else:
                    vals = [int(row)]
                vals = [x for x in vals if 0 <= x < len(self.examples)]
                if topk_keep > 0:
                    vals = vals[:topk_keep]
                if not vals:
                    vals = [int((len(cleaned) + 1) % len(self.examples))]
                cleaned.append(vals)
            self.self_mined_idx = cleaned
            logger.info("Loaded self-mined hard negatives: %d entries, first row K=%d", len(self.self_mined_idx), len(self.self_mined_idx[0]))

        if "train" in file_path:
            aug_nl_count = sum(getattr(x, 'has_aug_nl', 0) for x in self.examples)
            aug_code_count = sum(getattr(x, 'has_aug_code', 0) for x in self.examples)
            logger.info("ReFCode dataset: %d examples, aug_nl=%d, aug_code=%d", len(self.examples), aug_nl_count, aug_code_count)
            for idx, example in enumerate(self.examples[:3]):
                logger.info("*** Example ***")
                logger.info("idx: {}".format(idx))
                logger.info("code_tokens: {}".format([x.replace('Ġ','_') for x in example.code_tokens]))
                logger.info("code_ids: {}".format(' '.join(map(str, example.code_ids))))
                logger.info("nl_tokens: {}".format([x.replace('Ġ','_') for x in example.nl_tokens]))
                logger.info("nl_ids: {}".format(' '.join(map(str, example.nl_ids))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        ex = self.examples[i]
        if self.split == "train" and (getattr(self.args, 'use_refcode_uncertainty', False)
                                      or getattr(self.args, 'use_refcode_augmented_views', False)
                                      or getattr(self.args, 'use_refcode_global_hard_negative', False)):
            item = (
                torch.tensor(ex.code_ids),
                torch.tensor(ex.nl_ids),
                torch.tensor(ex.aug_nl_ids),
                torch.tensor(ex.has_aug_nl, dtype=torch.long),
                torch.tensor(ex.aug_code_ids),
                torch.tensor(ex.has_aug_code, dtype=torch.long),
                torch.tensor(i, dtype=torch.long),
            )
            if self.hard_idx is not None:
                hidx = int(self.hard_idx[i])
                if hidx < 0 or hidx >= len(self.examples):
                    raise IndexError(f"hard_idx[{i}]={hidx} out of range for {len(self.examples)} examples")
                item = item + (torch.tensor(self.examples[hidx].code_ids),)
            if self.self_mined_idx is not None:
                mids = self.self_mined_idx[i]
                mined_code_ids = [self.examples[int(mid)].code_ids for mid in mids]
                item = item + (torch.tensor(mined_code_ids, dtype=torch.long),)
            return item
        return (torch.tensor(ex.code_ids), torch.tensor(ex.nl_ids))
def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # all gpus
    torch.backends.cudnn.deterministic = True


def mask_tokens(inputs,tokenizer,mlm_probability):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, mlm_probability).to(inputs.device)
    special_tokens_mask = [tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in
                           labels.tolist()] # for masking special token
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool).to(inputs.device), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0) # masked padding
        
    masked_indices = torch.bernoulli(probability_matrix).bool() # will decide who will be masked
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool().to(inputs.device) & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool().to(inputs.device) & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long).to(inputs.device)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def replace_with_type_tokens(inputs,replaces,tokenizer,mlm_probability):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, mlm_probability).to(inputs.device)
    special_tokens_mask = [tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in
                           labels.tolist()] # for masking special token
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool).to(inputs.device), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0) # masked padding
        
    masked_indices = torch.bernoulli(probability_matrix).bool() # will decide who will be masked
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool().to(inputs.device) & masked_indices
    inputs[indices_replaced] = replaces[indices_replaced] 

    return inputs, labels

def replace_special_token_with_type_tokens(inputs, speical_token_ids, tokenizer, mlm_probability):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    labels = inputs.clone()
    probability_matrix = torch.full(labels.shape,0.0).to(inputs.device)   
    probability_matrix.masked_fill_(labels.eq(speical_token_ids).to(inputs.device), value=mlm_probability)
    masked_indices = torch.bernoulli(probability_matrix).bool() # will decide who will be masked
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool().to(inputs.device) & masked_indices
    inputs[indices_replaced] =  speical_token_ids

    return inputs, labels

def replace_special_token_with_mask(inputs, speical_token_ids, tokenizer, mlm_probability):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    labels = inputs.clone()
    probability_matrix = torch.full(labels.shape,0.0).to(inputs.device)   
    probability_matrix.masked_fill_(labels.eq(speical_token_ids).to(inputs.device), value=mlm_probability)
    masked_indices = torch.bernoulli(probability_matrix).bool() # will decide who will be masked
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool().to(inputs.device) & masked_indices
    inputs[indices_replaced] =tokenizer.convert_tokens_to_ids(tokenizer.mask_token) 

    return inputs, labels



def _tokenize_for_bm25(tokens):
    if tokens is None:
        return []
    if isinstance(tokens, str):
        text = tokens
    else:
        text = ' '.join([str(t) for t in tokens])
    return re.findall(r"[A-Za-z_][A-Za-z_0-9]*|\d+", text.lower())


def _bm25_scores_for_anchor(anchor_tokens, docs_tokens, k1=1.5, b=0.75):
    docs = [_tokenize_for_bm25(x) for x in docs_tokens]
    query = _tokenize_for_bm25(anchor_tokens)
    if not docs or not query:
        return [0.0 for _ in docs]
    n_docs = len(docs)
    avgdl = sum(len(d) for d in docs) / max(1, n_docs)
    df = Counter()
    for doc in docs:
        for tok in set(doc):
            df[tok] += 1
    scores = []
    for doc in docs:
        tf = Counter(doc)
        dl = len(doc)
        score = 0.0
        for tok in query:
            if tok not in tf:
                continue
            idf = math.log(1.0 + (n_docs - df.get(tok, 0) + 0.5) / (df.get(tok, 0) + 0.5))
            denom = tf[tok] + k1 * (1.0 - b + b * dl / max(avgdl, 1e-6))
            score += idf * (tf[tok] * (k1 + 1.0)) / max(denom, 1e-6)
        scores.append(score)
    return scores


def select_bm25_hard_negative_indices(nl_vec, raw_nl_tokens, urls=None, ratio=0.1, candidate_topk=-1):
    """ReFCode-style hard negative mining inside a batch."""
    with torch.no_grad():
        bs = nl_vec.size(0)
        if bs <= 1:
            return torch.zeros(bs, dtype=torch.long, device=nl_vec.device)
        sim = torch.matmul(nl_vec.detach(), nl_vec.detach().t())
        eye = torch.eye(bs, dtype=torch.bool, device=nl_vec.device)
        sim = sim.masked_fill(eye, -1e4)
        topk = bs - 1 if candidate_topk is None or candidate_topk <= 0 else min(candidate_topk, bs - 1)
        cand = torch.topk(sim, k=topk, dim=1).indices.cpu().tolist()
        hard = []
        urls = urls or [None] * bs
        for i in range(bs):
            cands = [j for j in cand[i] if j != i and (urls[i] is None or urls[j] != urls[i])]
            if not cands:
                cands = [j for j in range(bs) if j != i]
            cand_docs = [raw_nl_tokens[j] for j in cands]
            bm25_scores = _bm25_scores_for_anchor(raw_nl_tokens[i], cand_docs)
            ordered = sorted(zip(cands, bm25_scores), key=lambda x: x[1], reverse=True)
            rank = max(0, int(math.ceil(len(ordered) * ratio)) - 1)
            rank = min(rank, len(ordered) - 1)
            hard.append(ordered[rank][0])
        return torch.tensor(hard, dtype=torch.long, device=nl_vec.device)


def weighted_inter_modal_loss(
        query_vec,
        code_vec,
        temperature=0.03,
        hard_indices=None,
        hard_code_vec=None,
        hard_weight=1.0,
        hard_mode="batch_all",
        mask_self_in_hard_batch=True,
):
    """
    Inter-modal query->code contrastive loss.

    If hard_code_vec is provided, it should contain one offline global hard code
    representation for each query in the current batch. In batch_all mode, every
    query contrasts against the whole hard-code batch, giving [B, 2B] logits and
    B*B hard-negative pairs, matching the paper-style denominator.
    """
    bs = query_vec.size(0)
    base_logits = torch.matmul(query_vec, code_vec.t()) / temperature
    targets = torch.arange(bs, device=query_vec.device)
    if (hard_indices is None and hard_code_vec is None) or hard_weight <= 0:
        return F.cross_entropy(base_logits, targets)

    hard_mode = (hard_mode or "single").lower()
    hard_log_weight = math.log(max(float(hard_weight), 1e-8))
    if hard_code_vec is not None:
        hard_code_vec = hard_code_vec.to(query_vec.device)
    if hard_indices is not None:
        hard_indices = hard_indices.to(query_vec.device).long()

    if hard_mode == "single":
        pos = base_logits.diag()
        if hard_code_vec is not None:
            hard_logits = (query_vec * hard_code_vec).sum(dim=1) / temperature
        else:
            hard_logits = base_logits[targets, hard_indices]
        extra = hard_logits + hard_log_weight
        denom = torch.logsumexp(torch.cat([base_logits, extra.unsqueeze(1)], dim=1), dim=1)
        return -(pos - denom).mean()

    if hard_mode != "batch_all":
        raise ValueError(f"Unknown hard_mode={hard_mode}; expected 'single' or 'batch_all'.")

    if hard_code_vec is None:
        hard_code_vec = code_vec.index_select(0, hard_indices)
    hard_logits = torch.matmul(query_vec, hard_code_vec.t()) / temperature

    # Only possible in dynamic in-batch mode. Offline global hard negatives are
    # expected to be self/url-filtered by the builder.
    if mask_self_in_hard_batch and hard_indices is not None:
        self_positive_mask = hard_indices.unsqueeze(0).eq(targets.unsqueeze(1))
        hard_logits = hard_logits.masked_fill(self_positive_mask, -1e4)

    hard_logits = hard_logits + hard_log_weight
    logits_all = torch.cat([base_logits, hard_logits], dim=1)
    return F.cross_entropy(logits_all, targets)

def contrastive_pair_loss(anchor_vec, positive_vec, temperature=0.03, symmetric=True, valid_mask=None):
    logits = torch.matmul(anchor_vec, positive_vec.t()) / temperature
    targets = torch.arange(anchor_vec.size(0), device=anchor_vec.device)
    if valid_mask is None:
        loss = F.cross_entropy(logits, targets)
        if symmetric:
            loss = 0.5 * (loss + F.cross_entropy(logits.t(), targets))
        return loss
    valid_mask = valid_mask.float()
    per = F.cross_entropy(logits, targets, reduction='none')
    if valid_mask.sum() <= 0:
        return per.sum() * 0.0
    loss = (per * valid_mask).sum() / valid_mask.sum()
    if symmetric:
        per_t = F.cross_entropy(logits.t(), targets, reduction='none')
        loss_t = (per_t * valid_mask).sum() / valid_mask.sum()
        loss = 0.5 * (loss + loss_t)
    return loss


def gaussian_kl_loss(*encoded):
    losses = []
    for enc in encoded:
        mu, logvar = enc['mu'], enc['logvar']
        kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1)
        losses.append(kl.mean())
    if not losses:
        return torch.tensor(0.0)
    return sum(losses) / len(losses)



def _li_get_base_model(model):
    return model.module if hasattr(model, 'module') else model


def li_build_token_mask(input_ids, tokenizer=None):
    """Build token mask for lightweight late interaction."""
    pad_id = 1
    special_ids = set([0, 1, 2])

    if tokenizer is not None:
        try:
            pad_id = tokenizer.pad_token_id
            special_ids.add(tokenizer.cls_token_id)
            special_ids.add(tokenizer.sep_token_id)
            special_ids.add(tokenizer.pad_token_id)
            enc_only = tokenizer.convert_tokens_to_ids("<encoder-only>")
            if enc_only is not None and enc_only >= 0:
                special_ids.add(enc_only)
        except Exception:
            pass

    mask = input_ids.ne(pad_id)
    for sid in list(special_ids):
        if sid is not None and sid >= 0:
            mask = mask & input_ids.ne(int(sid))
    return mask


def li_encode_tokens(model, input_ids, tokenizer=None):
    """Return token-level hidden states from the underlying encoder."""
    base_model = _li_get_base_model(model)
    attention_mask = input_ids.ne(1)

    try:
        outputs = base_model.encoder(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state
    except TypeError:
        outputs = base_model.encoder(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )
        hidden = outputs[0]

    mask = li_build_token_mask(input_ids, tokenizer)
    return hidden, mask


def li_maxsim_score(q_hidden, q_mask, c_hidden, c_mask):
    """ColBERT-style MaxSim score."""
    q_hidden = F.normalize(q_hidden, p=2, dim=-1)
    c_hidden = F.normalize(c_hidden, p=2, dim=-1)

    sim = torch.bmm(q_hidden, c_hidden.transpose(1, 2))  # [N, Lq, Lc]
    sim = sim.masked_fill(~c_mask.unsqueeze(1), -1e4)

    max_sim = sim.max(dim=2).values  # [N, Lq]
    q_mask_f = q_mask.float()

    return (max_sim * q_mask_f).sum(dim=1) / q_mask_f.sum(dim=1).clamp_min(1.0)


def _li_truncate_inputs(input_ids, max_len):
    """Truncate only the lightweight LI branch input, keeping global branch unchanged."""
    if max_len is None or int(max_len) <= 0:
        return input_ids
    max_len = min(int(max_len), input_ids.size(1))
    return input_ids[:, :max_len].contiguous()


def compute_lite_late_interaction_self_mined_loss(model, tokenizer, nl_inputs, code_inputs, mined_code_inputs, args):
    """Lightweight training-stage late-interaction loss.

    This keeps the original global Stage2 loss unchanged:
      - BATCH_SIZE remains 128
      - in-batch negatives remain 128
      - self-mined negatives remain unchanged

    To reduce compute, only the extra LI branch is lightweight:
      1) randomly sample a subset of examples from the full batch;
      2) optionally truncate token length for LI branch only;
      3) optionally chunk the LI subset.
    """
    if mined_code_inputs is None or mined_code_inputs.dim() != 3:
        return nl_inputs.new_tensor(0.0, dtype=torch.float)

    bs, k, length = mined_code_inputs.size()
    if bs == 0 or k == 0:
        return nl_inputs.new_tensor(0.0, dtype=torch.float)

    # 1) Sample only a subset for the LI auxiliary branch.
    sample_size = int(getattr(args, 'li_sample_size', 16))
    if sample_size > 0 and sample_size < bs:
        idx = torch.randperm(bs, device=nl_inputs.device)[:sample_size]
        nl_inputs = nl_inputs.index_select(0, idx)
        code_inputs = code_inputs.index_select(0, idx)
        mined_code_inputs = mined_code_inputs.index_select(0, idx)
        bs = sample_size

    # 2) Use only a small number of mined negatives for LI.
    li_train_k = int(getattr(args, 'li_train_k', 1))
    if li_train_k <= 0:
        li_train_k = int(getattr(args, 'self_mined_train_k', 1))

    if li_train_k > 0 and k > li_train_k:
        perm = torch.randperm(k, device=mined_code_inputs.device)[:li_train_k]
        mined_code_inputs = mined_code_inputs.index_select(1, perm)
        k = li_train_k

    # 3) Truncate only the LI branch sequence length.
    li_nl_length = int(getattr(args, 'li_nl_length', 64))
    li_code_length = int(getattr(args, 'li_code_length', 128))

    nl_inputs = _li_truncate_inputs(nl_inputs, li_nl_length)
    code_inputs = _li_truncate_inputs(code_inputs, li_code_length)
    mined_code_inputs = mined_code_inputs[:, :, :min(li_code_length, mined_code_inputs.size(2))].contiguous()

    length = mined_code_inputs.size(2)

    chunk_size = int(getattr(args, 'li_chunk_size', 16))
    chunk_size = max(1, chunk_size)

    total_loss = nl_inputs.new_tensor(0.0, dtype=torch.float)
    total_count = 0

    for start in range(0, bs, chunk_size):
        end = min(start + chunk_size, bs)

        cur_nl_inputs = nl_inputs[start:end]
        cur_code_inputs = code_inputs[start:end]
        cur_mined_inputs = mined_code_inputs[start:end]

        cur_bs = cur_nl_inputs.size(0)

        q_hidden, q_mask = li_encode_tokens(model, cur_nl_inputs, tokenizer)
        pos_hidden, pos_mask = li_encode_tokens(model, cur_code_inputs, tokenizer)

        flat_mined = cur_mined_inputs.contiguous().view(cur_bs * k, length)
        mined_hidden, mined_mask = li_encode_tokens(model, flat_mined, tokenizer)
        mined_hidden = mined_hidden.view(cur_bs, k, mined_hidden.size(1), mined_hidden.size(2))
        mined_mask = mined_mask.view(cur_bs, k, mined_mask.size(1))

        pos_score = li_maxsim_score(q_hidden, q_mask, pos_hidden, pos_mask)

        q_rep = q_hidden.unsqueeze(1).expand(-1, k, -1, -1).contiguous().view(
            cur_bs * k, q_hidden.size(1), q_hidden.size(2)
        )
        q_mask_rep = q_mask.unsqueeze(1).expand(-1, k, -1).contiguous().view(
            cur_bs * k, q_mask.size(1)
        )

        neg_score = li_maxsim_score(
            q_rep,
            q_mask_rep,
            mined_hidden.contiguous().view(cur_bs * k, mined_hidden.size(2), mined_hidden.size(3)),
            mined_mask.contiguous().view(cur_bs * k, mined_mask.size(2)),
        ).view(cur_bs, k)

        tau = float(getattr(args, 'li_temperature', 0.05))
        tau = max(tau, 1e-6)

        logits = torch.cat([pos_score.unsqueeze(1), neg_score], dim=1) / tau
        targets = torch.zeros(cur_bs, dtype=torch.long, device=nl_inputs.device)

        chunk_loss = F.cross_entropy(logits, targets, reduction='sum')
        total_loss = total_loss + chunk_loss
        total_count += cur_bs

    return total_loss / max(total_count, 1)
def refcode_encode(model, input_ids, args):
    base_model = model.module if hasattr(model, 'module') else model
    return base_model.encode_inputs(input_ids, return_uncertainty=True, num_samples=args.refcode_uncertainty_samples)


def compute_ctrd_relevance_loss(model, nl_vec, code_vec, hard_code_vec, args):
    """Aggressive CTRD auxiliary objective.

    Positives:       (nl_i, code_i)
    Negatives:       (nl_i, offline_hard_code_i) when available
                     top-K most similar in-batch non-gold codes

    Loss = weighted BCE + pairwise ranking, so the relevance head not only
    classifies pairs but also explicitly forces positive code to score above
    difficult negatives. The encoder receives gradients from this loss.
    """
    base_model = model.module if hasattr(model, 'module') else model
    bs = nl_vec.size(0)
    device = nl_vec.device

    pos_logits = base_model.ctrd_logits(nl_vec, code_vec)  # [B]
    logits = [pos_logits]
    labels = [torch.ones_like(pos_logits)]
    weights = [torch.ones_like(pos_logits)]
    neg_logit_groups = []

    if hard_code_vec is not None and getattr(args, 'ctrd_hard_weight', 0.0) > 0:
        hard_logits = base_model.ctrd_logits(nl_vec, hard_code_vec.detach() if args.ctrd_detach_hard_code else hard_code_vec)
        logits.append(hard_logits)
        labels.append(torch.zeros_like(hard_logits))
        weights.append(torch.full_like(hard_logits, float(args.ctrd_hard_weight)))
        neg_logit_groups.append(hard_logits.view(bs, 1))

    topk = int(getattr(args, 'ctrd_batch_topk', 0))
    if topk > 0 and bs > 1 and getattr(args, 'ctrd_batch_weight', 0.0) > 0:
        with torch.no_grad():
            sim = torch.matmul(nl_vec.detach(), code_vec.detach().t())
            sim.fill_diagonal_(-1e4)
            k = min(topk, bs - 1)
            hard_idx = torch.topk(sim, k=k, dim=1).indices  # [B, K]

        flat_q = nl_vec.unsqueeze(1).expand(-1, k, -1).reshape(bs * k, -1)
        flat_c = code_vec.index_select(0, hard_idx.reshape(-1))
        batch_neg_logits = base_model.ctrd_logits(flat_q, flat_c)
        logits.append(batch_neg_logits)
        labels.append(torch.zeros_like(batch_neg_logits))
        weights.append(torch.full_like(batch_neg_logits, float(args.ctrd_batch_weight)))
        neg_logit_groups.append(batch_neg_logits.view(bs, k))

    all_logits = torch.cat(logits, dim=0)
    all_labels = torch.cat(labels, dim=0)
    all_weights = torch.cat(weights, dim=0)
    bce = F.binary_cross_entropy_with_logits(all_logits, all_labels, weight=all_weights)

    rank_loss = bce.new_tensor(0.0)
    if neg_logit_groups and getattr(args, 'ctrd_rank_weight', 0.0) > 0:
        margin = float(getattr(args, 'ctrd_rank_margin', 0.2))
        rank_terms = []
        for neg_logits in neg_logit_groups:
            rank_terms.append(F.softplus(neg_logits - pos_logits.unsqueeze(1) + margin).mean())
        rank_loss = sum(rank_terms) / len(rank_terms)

    return bce + float(args.ctrd_rank_weight) * rank_loss


def compute_self_mined_hard_loss(model, nl_vec, code_vec, mined_code_inputs, args):
    """Listwise loss over model-mined hard negatives.

    mined_code_inputs: [B, K, L], where each row contains codes that a previous
    checkpoint ranked highly for the same query but that are not the gold code.
    This directly optimizes the same cosine space used at retrieval time:
    gold code must beat the currently-confusing wrong codes.
    """
    if mined_code_inputs is None:
        return nl_vec.new_tensor(0.0)
    if mined_code_inputs.dim() != 3:
        return nl_vec.new_tensor(0.0)

    bs, k, length = mined_code_inputs.size()
    if bs == 0 or k == 0:
        return nl_vec.new_tensor(0.0)

    train_k = int(getattr(args, 'self_mined_train_k', 0))
    if train_k > 0 and k > train_k:
        perm = torch.randperm(k, device=mined_code_inputs.device)[:train_k]
        mined_code_inputs = mined_code_inputs.index_select(1, perm)
        k = train_k

    flat = mined_code_inputs.contiguous().view(bs * k, length)
    if getattr(args, 'use_refcode_uncertainty', False):
        mined_enc = refcode_encode(model, flat, args)
        mined_vec = mined_enc['z']
    else:
        mined_vec = model(code_inputs=flat)
    mined_vec = mined_vec.view(bs, k, -1)

    pos_sim = torch.einsum('bd,bd->b', nl_vec, code_vec)
    neg_sim = torch.einsum('bd,bkd->bk', nl_vec, mined_vec)

    fn_margin = float(getattr(args, 'self_mined_false_negative_margin', 999.0))
    if fn_margin < 100.0:
        keep = neg_sim <= (pos_sim.unsqueeze(1) + fn_margin)
        row_has = keep.any(dim=1)
        if not row_has.all():
            hardest = neg_sim.argmax(dim=1)
            keep[~row_has, :] = False
            keep[~row_has, hardest[~row_has]] = True
        neg_sim = neg_sim.masked_fill(~keep, -1e4)

    tau = float(getattr(args, 'self_mined_temperature', 0.03))
    tau = max(tau, 1e-6)
    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1) / tau
    targets = torch.zeros(bs, dtype=torch.long, device=nl_vec.device)
    ce_loss = F.cross_entropy(logits, targets)

    max_weight = float(getattr(args, 'self_mined_max_weight', 0.2))
    if max_weight > 0:
        hardest_neg = neg_sim.max(dim=1).values
        margin = float(getattr(args, 'self_mined_margin', 0.02))
        max_loss = F.softplus((hardest_neg - pos_sim + margin) / tau).mean()
        ce_loss = ce_loss + max_weight * max_loss
    return ce_loss


def train(args, model, tokenizer,pool):

    """Train the model. With --use_refcode_uncertainty, use ReFCode-style loss."""
    if args.data_aug_type ==  "replace_type" :
        train_dataset=TextDataset(tokenizer, args, args.train_data_file, pool)
    else:
        train_dataset=TextDataset_unixcoder(tokenizer, args, args.train_data_file, pool)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,num_workers=4,drop_last=True)

    model.to(args.device)
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.num_warmup_steps, num_training_steps=len(train_dataloader)*args.num_train_epochs)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Num quene = %d", args.moco_k)
    logger.info("  Instantaneous batch size per device = %d", args.train_batch_size // max(args.n_gpu, 1))
    logger.info("  Total train batch size  = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", len(train_dataloader)*args.num_train_epochs)
    logger.info("  ReFCode enabled = %s", args.use_refcode_uncertainty)
    logger.info("  ReFCode hard negative enabled = %s", args.use_refcode_global_hard_negative)
    logger.info("  ReFCode hard mode = %s", args.refcode_hard_mode)
    logger.info("  ReFCode offline hard idx = %s", getattr(args, 'refcode_hard_idx_file', ''))
    logger.info("  CTRD enabled = %s", getattr(args, 'use_ctrd', False))
    logger.info("  CTRD weight/topk/hard_w/rank_w = %s/%s/%s/%s",
                getattr(args, 'ctrd_weight', 0.0), getattr(args, 'ctrd_batch_topk', 0),
                getattr(args, 'ctrd_hard_weight', 0.0), getattr(args, 'ctrd_rank_weight', 0.0))
    logger.info("  Self-mined hard negative enabled = %s", getattr(args, 'use_self_mined_hard_negative', False))
    logger.info("  Self-mined idx/topk/train_k/weight = %s/%s/%s/%s",
                getattr(args, 'self_mined_idx_file', ''), getattr(args, 'self_mined_topk', 0),
                getattr(args, 'self_mined_train_k', 0), getattr(args, 'self_mined_weight', 0.0))
    logger.info("  Lite late interaction enabled/weight/temp/train_k/sample/every/chunk/len = %s/%s/%s/%s/%s/%s/%s/%s-%s",
                getattr(args, 'use_lite_late_interaction_train', 0), getattr(args, 'li_weight', 0.0),
                getattr(args, 'li_temperature', 0.05), getattr(args, 'li_train_k', 1),
                getattr(args, 'li_sample_size', 16), getattr(args, 'li_every_n_steps', 2),
                getattr(args, 'li_chunk_size', 16), getattr(args, 'li_nl_length', 64),
                getattr(args, 'li_code_length', 128))

    model.zero_grad()
    model.train()
    tr_num,tr_loss,best_mrr=0,0,-1
    loss_fct = CrossEntropyLoss()

    for idx in range(args.num_train_epochs):
        for step,batch in enumerate(train_dataloader):
            code_inputs = batch[0].to(args.device)
            nl_inputs = batch[1].to(args.device)
            targets = torch.arange(code_inputs.size(0), device=code_inputs.device)

            if not args.use_refcode_uncertainty:
                code_vec = model(code_inputs=code_inputs)
                nl_vec = model(nl_inputs=nl_inputs)
                scores = torch.einsum("ab,cb->ac",nl_vec,code_vec)
                loss = loss_fct(scores*20, targets)
            else:
                code_enc = refcode_encode(model, code_inputs, args)
                nl_enc = refcode_encode(model, nl_inputs, args)
                code_vec, nl_vec = code_enc['z'], nl_enc['z']

                hard_indices = None
                hard_code_vec = None
                if args.use_refcode_global_hard_negative:
                    if getattr(args, 'refcode_hard_idx_file', '') and len(batch) >= 8:
                        # Paper-style path: offline global TopK+BM25 hard code ids.
                        hard_code_inputs = batch[7].to(args.device)
                        hard_code_enc = refcode_encode(model, hard_code_inputs, args)
                        hard_code_vec = hard_code_enc['z']
                    else:
                        # Fallback: dynamic batch-local cosine+BM25 mining.
                        batch_indices = batch[6].cpu().tolist() if len(batch) >= 7 else list(range(code_inputs.size(0)))
                        raw_nl = [train_dataset.examples[i].raw_nl_tokens for i in batch_indices]
                        urls = [train_dataset.examples[i].url for i in batch_indices]
                        hard_indices = select_bm25_hard_negative_indices(
                            nl_vec, raw_nl, urls=urls,
                            ratio=args.refcode_hn_bm25_rank_ratio,
                            candidate_topk=args.refcode_hn_candidate_topk,
                        )

                mined_code_inputs = None
                if getattr(args, 'use_self_mined_hard_negative', False) and len(batch) >= 8:
                    maybe_mined = batch[-1]
                    if hasattr(maybe_mined, 'dim') and maybe_mined.dim() == 3:
                        mined_code_inputs = maybe_mined.to(args.device)

                loss_inter = weighted_inter_modal_loss(
                    nl_vec, code_vec, temperature=args.refcode_temperature,
                    hard_indices=hard_indices,
                    hard_code_vec=hard_code_vec,
                    hard_weight=args.refcode_hard_weight if args.use_refcode_global_hard_negative else 0.0,
                    hard_mode=args.refcode_hard_mode,
                    mask_self_in_hard_batch=not args.refcode_no_mask_self_in_hard_batch,
                )
                if args.refcode_symmetric_inter:
                    loss_inter_rev = weighted_inter_modal_loss(code_vec, nl_vec, temperature=args.refcode_temperature)
                    loss_inter = 0.5 * (loss_inter + loss_inter_rev)

                loss_intra_nl = contrastive_pair_loss(nl_enc['z'], nl_enc['z_plus'], temperature=args.refcode_temperature, symmetric=True)
                loss_intra_code = contrastive_pair_loss(code_enc['z'], code_enc['z_plus'], temperature=args.refcode_temperature, symmetric=True)
                loss_intra = 0.5 * (loss_intra_nl + loss_intra_code)
                loss_kl = gaussian_kl_loss(nl_enc, code_enc)
                loss = loss_inter + args.refcode_intra_weight * loss_intra + args.refcode_kl_weight * loss_kl

                if getattr(args, 'use_ctrd', False) and args.ctrd_weight > 0:
                    loss_ctrd = compute_ctrd_relevance_loss(
                        model=model,
                        nl_vec=nl_vec,
                        code_vec=code_vec,
                        hard_code_vec=hard_code_vec if args.ctrd_use_offline_hard else None,
                        args=args,
                    )
                    loss = loss + args.ctrd_weight * loss_ctrd

                if (getattr(args, 'use_self_mined_hard_negative', False)
                        and getattr(args, 'self_mined_weight', 0.0) > 0
                        and mined_code_inputs is not None):
                    loss_self_mined = compute_self_mined_hard_loss(
                        model=model,
                        nl_vec=nl_vec,
                        code_vec=code_vec,
                        mined_code_inputs=mined_code_inputs,
                        args=args,
                    )
                    loss = loss + args.self_mined_weight * loss_self_mined

                li_every_n_steps = max(1, int(getattr(args, 'li_every_n_steps', 1)))
                if (getattr(args, 'use_lite_late_interaction_train', 0)
                        and getattr(args, 'li_weight', 0.0) > 0
                        and mined_code_inputs is not None
                        and ((step + 1) % li_every_n_steps == 0)):
                    loss_li = compute_lite_late_interaction_self_mined_loss(
                        model=model,
                        tokenizer=tokenizer,
                        nl_inputs=nl_inputs,
                        code_inputs=code_inputs,
                        mined_code_inputs=mined_code_inputs,
                        args=args,
                    )
                    loss = loss + float(args.li_weight) * loss_li

                if args.use_refcode_augmented_views and len(batch) >= 6:
                    aug_nl_inputs = batch[2].to(args.device)
                    has_aug_nl = batch[3].to(args.device).float()
                    aug_code_inputs = batch[4].to(args.device)
                    has_aug_code = batch[5].to(args.device).float()

                    if has_aug_nl.sum() > 0 and args.refcode_aug_nl_weight > 0:
                        aug_nl_enc = refcode_encode(model, aug_nl_inputs, args)
                        loss_aug_nl_search = contrastive_pair_loss(aug_nl_enc['z'], code_vec, temperature=args.refcode_temperature, symmetric=False, valid_mask=has_aug_nl)
                        loss_aug_nl_cons = contrastive_pair_loss(nl_vec, aug_nl_enc['z'], temperature=args.refcode_temperature, symmetric=True, valid_mask=has_aug_nl)
                        loss = loss + args.refcode_aug_nl_weight * (loss_aug_nl_search + 0.5 * loss_aug_nl_cons)

                    if has_aug_code.sum() > 0 and args.refcode_aug_code_weight > 0:
                        aug_code_enc = refcode_encode(model, aug_code_inputs, args)
                        loss_aug_code_search = contrastive_pair_loss(nl_vec, aug_code_enc['z'], temperature=args.refcode_temperature, symmetric=False, valid_mask=has_aug_code)
                        loss_aug_code_cons = contrastive_pair_loss(code_vec, aug_code_enc['z'], temperature=args.refcode_temperature, symmetric=True, valid_mask=has_aug_code)
                        loss = loss + args.refcode_aug_code_weight * (loss_aug_code_search + 0.5 * loss_aug_code_cons)

            tr_loss += loss.item()
            tr_num += 1

            if (step+1)% args.eval_frequency==0:
                logger.info("epoch {} step {} loss {}".format(idx,step+1,round(tr_loss/max(1,tr_num),5)))
                tr_loss=0
                tr_num=0

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        results = evaluate(args, model, tokenizer,args.eval_data_file, pool, eval_when_training=True)
        for key, value in results.items():
            logger.info("  %s = %s", key, round(value,4) if isinstance(value, float) else value)

        select_mrr = results['eval_mrr']
        select_name = 'eval_mrr'

        if getattr(args, 'use_valid_fusion_select', 0):
            fusion_results = evaluate_late_fusion(
                args,
                model,
                tokenizer,
                args.eval_data_file,
                pool,
                eval_when_training=True,
            )
            for key, value in fusion_results.items():
                logger.info("  %s = %s", key, round(value,4) if isinstance(value, float) else value)

            select_mrr = fusion_results['eval_fusion_mrr']
            select_name = 'eval_fusion_mrr'

        if select_mrr > best_mrr:
            best_mrr = select_mrr
            logger.info("  "+"*"*20)
            logger.info("  Best %s:%s", select_name, round(best_mrr,4))
            logger.info("  "+"*"*20)
            checkpoint_prefix = 'checkpoint-best-mrr'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = model.module if hasattr(model,'module') else model
            output_dir = os.path.join(output_dir, '{}'.format('model.bin'))
            torch.save(model_to_save.state_dict(), output_dir)
            logger.info("Saving model checkpoint to %s", output_dir)

def  multi_lang_continue_pre_train(args, model, tokenizer,pool):
    """ Train the model """
    #get training dataset
    if "unixcoder" in args.model_name_or_path:
        train_datasets = []
        for train_data_file in args.couninue_pre_train_data_files:
            train_dataset=TextDataset_unixcoder(tokenizer, args, train_data_file, pool)
            train_datasets.append(train_dataset)
    else:
        train_datasets = []
        for train_data_file in args.couninue_pre_train_data_files:
            train_dataset=TextDataset(tokenizer, args, train_data_file, pool)
            train_datasets.append(train_dataset)

    train_samplers = [RandomSampler(train_dataset) for train_dataset in train_datasets]
    # https://blog.csdn.net/weixin_44966641/article/details/124878064
    train_dataloaders = [cycle(DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,drop_last=True)) for train_dataset,train_sampler in zip(train_datasets,train_samplers)]
    t_total = args.max_steps

    #get optimizer and scheduler
    # Prepare optimizer and schedule (linear warmup and decay)https://huggingface.co/transformers/v3.3.1/training.html
    model.to(args.device)
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.num_warmup_steps,num_training_steps=t_total)

    # Train!
    training_data_length = sum ([len(item) for item in train_datasets])
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", training_data_length)
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Num quene = %d", args.moco_k)
    logger.info("  Instantaneous batch size per device = %d", args.train_batch_size // max(args.n_gpu, 1))
    logger.info("  Total train batch size  = %d", args.train_batch_size)

    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    scheduler_last = os.path.join(checkpoint_last, 'scheduler.pt')
    optimizer_last = os.path.join(checkpoint_last, 'optimizer.pt')
    if os.path.exists(scheduler_last):
        scheduler.load_state_dict(torch.load(scheduler_last, map_location="cpu"))
    if os.path.exists(optimizer_last):
        optimizer.load_state_dict(torch.load(optimizer_last, map_location="cpu"))    
    if args.local_rank == 0:
        torch.distributed.barrier()         
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install NVIDIA Apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank%args.gpu_per_node],
                                                          output_device=args.local_rank%args.gpu_per_node,
                                                          find_unused_parameters=True)
 
    loss_fct = CrossEntropyLoss()
    set_seed(args.seed)  # Added here for reproducibility (even between python 2 and 3)
    probs=[len(x) for x in train_datasets]
    probs=[x/sum(probs) for x in probs]
    probs=[x**0.7 for x in probs]
    probs=[x/sum(probs) for x in probs]
    # global_step = args.start_step
    model.zero_grad()
    model.train()

    global_step = args.start_step
    step=0
    tr_loss, logging_loss,avg_loss,tr_nb, best_mrr = 0.0, 0.0,0.0,0,-1
    tr_num=0
    special_token_list = all_special_token 
    special_token_id_list = tokenizer.convert_tokens_to_ids(special_token_list)
    while True: 
        
        train_dataloader=np.random.choice(train_dataloaders, 1, p=probs)[0]
        # train_dataloader=train_dataloader[0]
        step+=1
        batch=next(train_dataloader)
        # source_ids= batch.to(args.device)
        model.train()
        # loss = model(source_ids)
        code_inputs = batch[0].to(args.device)  
        code_transformations_ids = code_inputs.clone()
        nl_inputs = batch[1].to(args.device)
        nl_transformations_ids= nl_inputs.clone()
        
        if step%4 == 0:
            code_transformations_ids[:, 3:], _ = mask_tokens(code_inputs.clone()[:, 3:] ,tokenizer,args.mlm_probability)
            nl_transformations_ids[:, 3:], _ = mask_tokens(nl_inputs.clone()[:, 3:] ,tokenizer,args.mlm_probability)
        elif step%4 == 1:
            code_types = code_inputs.clone()
            code_transformations_ids[:, 3:], _ = replace_with_type_tokens(code_inputs.clone()[:, 3:] ,code_types.clone()[:, 3:],tokenizer,args.mlm_probability)
        elif step%4 == 2:
            random.seed( step)
            choice_token_id  = choice(special_token_id_list)
            code_transformations_ids[:, 3:], _ = replace_special_token_with_type_tokens(code_inputs.clone()[:, 3:], choice_token_id, tokenizer,args.mlm_probability)
        elif step%4 == 3:
            random.seed( step)
            choice_token_id  = choice(special_token_id_list)
            code_transformations_ids[:, 3:], _ = replace_special_token_with_mask(code_inputs.clone()[:, 3:], choice_token_id, tokenizer,args.mlm_probability)
        

        tr_num+=1   
        inter_output, inter_target, _, _= model(source_code_q=code_inputs, source_code_k=code_transformations_ids, 
                                    nl_q=nl_inputs , nl_k=nl_transformations_ids )
        
        
        
        # loss_fct = CrossEntropyLoss()
        loss = loss_fct(20*inter_output, inter_target)

        if args.n_gpu > 1:
            loss = loss.mean() # mean() to average on multi-gpu parallel training

            
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        tr_loss += loss.item()
        if (step+1)% args.eval_frequency==0:
            logger.info("step {} loss {}".format(step+1,round(tr_loss/tr_num,5)))
            tr_loss=0
            tr_num=0

        if (step + 1) % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()  
            global_step += 1
            output_flag=True
            avg_loss=round((tr_loss - logging_loss) /(global_step- tr_nb),6)

            if global_step %100 == 0:
                logger.info(" global steps (step*gradient_accumulation_steps ): %s loss: %s", global_step, round(avg_loss,6))
            if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                logging_loss = tr_loss
                tr_nb=global_step

            if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                checkpoint_prefix = 'checkpoint-mrr'
                # results = evaluate(args, model, tokenizer,pool=pool,eval_when_training=True)
                results = evaluate(args, model, tokenizer,args.eval_data_file, pool, eval_when_training=True)

                # for key, value in results.items():
                #     logger.info("  %s = %s", key, round(value,6))
                logger.info("  %s = %s", 'eval_mrr', round(results['eval_mrr'],6))

                if results['eval_mrr']>best_mrr:
                    best_mrr=results['eval_mrr']
                    logger.info("  "+"*"*20)  
                    logger.info("  Best mrr:%s",round(best_mrr,4))
                    logger.info("  "+"*"*20)                          

                    output_dir = os.path.join(args.output_dir, '{}'.format('checkpoint-best-mrr'))                        
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)                        
                    model_to_save = model.module if hasattr(model,'module') else model
                    output_dir = os.path.join(output_dir, '{}'.format('model.bin')) 
                    torch.save(model_to_save.state_dict(), output_dir)
                    logger.info("Saving model checkpoint to %s", output_dir)



                # Save model checkpoint
                output_dir = os.path.join(args.output_dir, '{}-{}-{}'.format(checkpoint_prefix, global_step,round(results['eval_mrr'],6)))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = model.module.code_encoder_q  if hasattr(model,'module') else model.code_encoder_q   # Take care of distributed/parallel training
                model_to_save.save_pretrained(output_dir)
                torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                logger.info("Saving model checkpoint to %s", output_dir)

                # _rotate_checkpoints(args, checkpoint_prefix)

                last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
                if not os.path.exists(last_output_dir):
                    os.makedirs(last_output_dir)
                model_to_save.save_pretrained(last_output_dir)
                idx_file = os.path.join(last_output_dir, 'idx_file.txt')
                with open(idx_file, 'w', encoding='utf-8') as idxf:
                    idxf.write(str(0) + '\n')

                torch.save(optimizer.state_dict(), os.path.join(last_output_dir, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(last_output_dir, "scheduler.pt"))
                logger.info("Saving optimizer and scheduler states to %s", last_output_dir)

                step_file = os.path.join(last_output_dir, 'step_file.txt')
                with open(step_file, 'w', encoding='utf-8') as stepf:
                    stepf.write(str(global_step) + '\n')

            if args.max_steps > 0 and global_step > args.max_steps:
                break



def _vf_zscore(x):
    x = np.asarray(x, dtype=np.float32)
    return (x - x.mean()) / (x.std() + 1e-6)


def evaluate_late_fusion(args, model, tokenizer, file_name, pool, eval_when_training=False):
    """Evaluate fixed global-local fusion MRR on validation/test split.

    This function is used to select checkpoint by validation fusion MRR.

    Protocol:
      1. encode all queries/codes with global bi-encoder;
      2. use global score to retrieve topK candidates;
      3. compute token-level late-interaction score only inside topK;
      4. fuse normalized global score and local score with fixed alpha;
      5. compute MRR.

    It does NOT train on validation/test labels.
    It is only used for checkpoint selection on valid set and final evaluation.
    """
    dataset_class = TextDataset_unixcoder
    query_dataset = dataset_class(tokenizer, args, file_name, pool)
    code_dataset = dataset_class(tokenizer, args, args.codebase_file, pool)

    query_sampler = SequentialSampler(query_dataset)
    code_sampler = SequentialSampler(code_dataset)

    query_dataloader = DataLoader(
        query_dataset,
        sampler=query_sampler,
        batch_size=args.eval_batch_size,
        num_workers=4,
    )
    code_dataloader = DataLoader(
        code_dataset,
        sampler=code_sampler,
        batch_size=args.eval_batch_size,
        num_workers=4,
    )

    logger.info("***** Running late-fusion evaluation on %s *****" % args.lang)
    logger.info("  Num queries = %d", len(query_dataset))
    logger.info("  Num codes = %d", len(code_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    logger.info("  Fusion topK/alpha = %s/%s",
                getattr(args, 'valid_fusion_topk', 50),
                getattr(args, 'valid_fusion_alpha', 0.5))

    model.eval()

    # 1) pooled global embeddings
    code_vecs = []
    nl_vecs = []

    for batch in query_dataloader:
        nl_inputs = batch[-1].to(args.device)
        with torch.no_grad():
            nl_vec = model(nl_inputs=nl_inputs)
        nl_vecs.append(nl_vec.detach().float().cpu().numpy())

    for batch in code_dataloader:
        code_inputs = batch[0].to(args.device)
        with torch.no_grad():
            code_vec = model(code_inputs=code_inputs)
        code_vecs.append(code_vec.detach().float().cpu().numpy())

    code_vecs = np.concatenate(code_vecs, 0)
    nl_vecs = np.concatenate(nl_vecs, 0)

    scores = np.matmul(nl_vecs, code_vecs.T)

    nl_urls = [example.url for example in query_dataset.examples]
    code_urls = [example.url for example in code_dataset.examples]

    url_to_code_idx = {}
    for idx, url in enumerate(code_urls):
        if url not in url_to_code_idx:
            url_to_code_idx[url] = idx

    top_k = int(getattr(args, 'valid_fusion_topk', 50))
    alpha = float(getattr(args, 'valid_fusion_alpha', 0.5))
    rerank_batch_size = int(getattr(args, 'valid_rerank_batch_size', 64))
    use_fp16 = bool(int(getattr(args, 'valid_fusion_fp16', 1)))

    bi_ranks = []
    li_ranks = []
    fusion_ranks = []

    for qi in range(len(query_dataset)):
        gold_idx = url_to_code_idx.get(nl_urls[qi], None)
        if gold_idx is None:
            bi_ranks.append(0)
            li_ranks.append(0)
            fusion_ranks.append(0)
            continue

        query_scores = scores[qi]
        gold_score = query_scores[gold_idx]
        bi_rank = int(np.sum(query_scores > gold_score)) + 1
        bi_ranks.append(1.0 / bi_rank)

        k = min(top_k, query_scores.shape[0])
        # partial topK then sort by score desc
        top_idx = np.argpartition(-query_scores, kth=k - 1)[:k]
        top_idx = top_idx[np.argsort(-query_scores[top_idx])]
        bi_top_scores = query_scores[top_idx]

        # if gold is outside topK, reranker cannot recover it; use original bi-rank.
        if gold_idx not in set(top_idx.tolist()):
            li_ranks.append(1.0 / bi_rank)
            fusion_ranks.append(1.0 / bi_rank)
            continue

        q_ids = torch.tensor(query_dataset.examples[qi].nl_ids, dtype=torch.long, device=args.device).unsqueeze(0)
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=use_fp16 and args.device.type == "cuda"):
                q_hidden, q_mask = li_encode_tokens(model, q_ids, tokenizer)
            q_hidden = q_hidden.repeat(k, 1, 1)
            q_mask = q_mask.repeat(k, 1)

        li_scores_all = []
        for start in range(0, k, rerank_batch_size):
            end = min(start + rerank_batch_size, k)
            cand_ids = [code_dataset.examples[int(cid)].code_ids for cid in top_idx[start:end]]
            c_ids = torch.tensor(cand_ids, dtype=torch.long, device=args.device)

            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=use_fp16 and args.device.type == "cuda"):
                    c_hidden, c_mask = li_encode_tokens(model, c_ids, tokenizer)
                local_scores = li_maxsim_score(
                    q_hidden[start:end].float(),
                    q_mask[start:end],
                    c_hidden.float(),
                    c_mask,
                )
                li_scores_all.append(local_scores.detach().float().cpu())

        li_scores = torch.cat(li_scores_all, dim=0).numpy()

        li_order = np.argsort(-li_scores)
        li_ids = top_idx[li_order]
        li_pos = int(np.where(li_ids == gold_idx)[0][0]) + 1
        li_ranks.append(1.0 / li_pos)

        final_scores = alpha * _vf_zscore(bi_top_scores) + (1.0 - alpha) * _vf_zscore(li_scores)
        fusion_order = np.argsort(-final_scores)
        fusion_ids = top_idx[fusion_order]
        fusion_pos = int(np.where(fusion_ids == gold_idx)[0][0]) + 1
        fusion_ranks.append(1.0 / fusion_pos)

    model.train()

    result = {}
    result["eval_bi_mrr"] = float(np.mean(bi_ranks))
    result["eval_li_only_mrr"] = float(np.mean(li_ranks))
    result["eval_fusion_mrr"] = float(np.mean(fusion_ranks))
    result["fusion_topk"] = top_k
    result["fusion_alpha"] = alpha
    return result


def evaluate(args, model, tokenizer,file_name,pool, eval_when_training=False):
    # if "unixcoder" in args.model_name_or_path or "coco" in args.model_name_or_path :
    dataset_class = TextDataset_unixcoder
    # else:
        # dataset_class = TextDataset
    query_dataset = dataset_class(tokenizer, args, file_name, pool)
    query_sampler = SequentialSampler(query_dataset)
    query_dataloader = DataLoader(query_dataset, sampler=query_sampler, batch_size=args.eval_batch_size,num_workers=4)
    
    code_dataset = dataset_class(tokenizer, args, args.codebase_file, pool)
    code_sampler = SequentialSampler(code_dataset)
    code_dataloader = DataLoader(code_dataset, sampler=code_sampler, batch_size=args.eval_batch_size,num_workers=4)    

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation on %s *****"%args.lang)
    logger.info("  Num queries = %d", len(query_dataset))
    logger.info("  Num codes = %d", len(code_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    
    model.eval()
    model_eval = model.module if hasattr(model,'module') else model
    code_vecs=[] 
    nl_vecs=[]
    for batch in query_dataloader:  
        nl_inputs = batch[-1].to(args.device)
        with torch.no_grad():
            if args.model_type ==  "base" :
                nl_vec = model(nl_inputs=nl_inputs) 

            elif args.model_type in  ["refcode" ,"no_aug_refcode", "multi-loss-refcode"]:
                outputs = model_eval.nl_encoder_q(nl_inputs, attention_mask=nl_inputs.ne(1))
                if args.agg_way == "avg":
                    outputs = outputs [0]
                    nl_vec = (outputs*nl_inputs.ne(1)[:,:,None]).sum(1)/nl_inputs.ne(1).sum(-1)[:,None] # None作为ndarray或tensor的索引作用是增加维度，
                elif args.agg_way == "cls_pooler":
                    nl_vec =outputs [1]
                elif args.agg_way == "avg_cls_pooler":
                     nl_vec =outputs [1] +  (outputs[0]*nl_inputs.ne(1)[:,:,None]).sum(1)/nl_inputs.ne(1).sum(-1)[:,None] 
                nl_vec  = torch.nn.functional.normalize( nl_vec, p=2, dim=1)
                if args.do_whitening:
                    nl_vec=whitening_torch_final(nl_vec)


            
            nl_vecs.append(nl_vec.cpu().numpy()) 

    for batch in code_dataloader:
        with torch.no_grad():
            code_inputs = batch[0].to(args.device)
            if args.model_type ==  "base" :
                code_vec = model(code_inputs=code_inputs)
            elif args.model_type in  ["refcode" ,"no_aug_refcode", "multi-loss-refcode"]:
                # code_vec =  model_eval.code_encoder_q(code_inputs, attention_mask=code_inputs.ne(1))[1]
                outputs = model_eval.code_encoder_q(code_inputs, attention_mask=code_inputs.ne(1))
                if args.agg_way == "avg":
                    outputs = outputs [0]
                    code_vec  = (outputs*code_inputs.ne(1)[:,:,None]).sum(1)/code_inputs.ne(1).sum(-1)[:,None] # None作为ndarray或tensor的索引作用是增加维度，
                elif args.agg_way == "cls_pooler":
                    code_vec=outputs [1]
                elif args.agg_way == "avg_cls_pooler":
                     code_vec=outputs [1] +  (outputs[0]*code_inputs.ne(1)[:,:,None]).sum(1)/code_inputs.ne(1).sum(-1)[:,None] 
                code_vec  = torch.nn.functional.normalize(code_vec, p=2, dim=1)
                if args.do_whitening:
                    code_vec=whitening_torch_final(code_vec)
        
            
            
            code_vecs.append(code_vec.cpu().numpy())  

    model.train()    
    code_vecs=np.concatenate(code_vecs,0)
    nl_vecs=np.concatenate(nl_vecs,0)

    scores=np.matmul(nl_vecs,code_vecs.T)
    
    sort_ids=np.argsort(scores, axis=-1, kind='quicksort', order=None)[:,::-1]    
    
    nl_urls=[]
    code_urls=[]
    for example in query_dataset.examples:
        nl_urls.append(example.url)
        
    for example in code_dataset.examples:
        code_urls.append(example.url)
        
    ranks=[]
    for url, sort_id in zip(nl_urls,sort_ids):
        rank=0
        find=False
        for idx in sort_id[:1000]:
            if find is False:
                rank+=1
            if code_urls[idx]==url:
                find=True
        if find:
            ranks.append(1/rank)
        else:
            ranks.append(0)
    if args.save_evaluation_reuslt:
        evaluation_result = {"nl_urls":nl_urls, "code_urls":code_urls,"sort_ids":sort_ids[:,:10],"ranks":ranks}
        save_pickle_data(args.save_evaluation_reuslt_dir, "evaluation_result.pkl",evaluation_result)
    result = cal_r1_r5_r10(ranks)
    result["eval_mrr"]  = float(np.mean(ranks))
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    # soda
    parser.add_argument('--data_aug_type',default="replace_type",choices=["replace_type", "random_mask" ,"other"], help="the ways of soda",required=False)
    parser.add_argument('--aug_type_way',default="random_replace_type",choices=["random_replace_type", "replace_special_type" ,"replace_special_type_with_mask"], help="the ways of soda",required=False)
    parser.add_argument('--print_align_unif_loss', action='store_true', help='print_align_unif_loss', required=False)
    parser.add_argument('--do_ineer_loss', action='store_true', help='print_align_unif_loss', required=False)
    parser.add_argument('--only_save_the_nl_code_vec', action='store_true', help='print_align_unif_loss', required=False)
    parser.add_argument('--do_zero_short', action='store_true', help='print_align_unif_loss', required=False)
    parser.add_argument('--agg_way',default="avg",choices=["avg", "cls_pooler","avg_cls_pooler" ], help="base is codebert/graphcoder/unixcoder",required=False)
    parser.add_argument('--weight_decay',default=0.01, type=float,required=False)
    parser.add_argument('--do_single_lang_continue_pre_train', action='store_true', help='do_single_lang_continue_pre_train', required=False)
    parser.add_argument('--save_evaluation_reuslt', action='store_true', help='save_evaluation_reuslt', required=False)
    parser.add_argument('--save_evaluation_reuslt_dir', type=str, help='save_evaluation_reuslt', required=False)
    parser.add_argument('--epoch', type=int, default=50,
                        help="random seed for initialization")
    # new continue pre-training
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--loaded_model_filename", type=str, required=False,
                        help="loaded_model_filename")
    parser.add_argument("--loaded_codebert_model_filename", type=str, required=False,
                        help="loaded_model_filename")
    parser.add_argument('--do_multi_lang_continue_pre_train', action='store_true', help='do_multi_lang_continue_pre_train', required=False)
    parser.add_argument("--couninue_pre_train_data_files", default=["dataset/ruby/train.jsonl",  "dataset/java/train.jsonl",], type=str, nargs='+', required=False,
                        help="The input training data files (some json files).")
    # parser.add_argument("--couninue_pre_train_data_files", default=["dataset/go/train.jsonl",  "dataset/java/train.jsonl",
    # "dataset/javascript/train.jsonl",  "dataset/php/train.jsonl",  "dataset/python/train.jsonl",  "dataset/ruby/train.jsonl",], type=list, required=False,
    #                     help="The input training data files (some json files).")
    parser.add_argument('--do_continue_pre_trained', action='store_true', help='debug mode', required=False)
    parser.add_argument('--do_fine_tune', action='store_true', help='debug mode', required=False)
    parser.add_argument('--do_whitening', action='store_true', help='do_whitening', required=False)
    parser.add_argument("--time_score", default=1, type=int,help="cosine value * time_score")   
    parser.add_argument("--max_steps", default=100, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--num_warmup_steps", default=0, type=int, help="num_warmup_steps")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")    
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    # new moco
    parser.add_argument('--moco_type',default="encoder_queue",choices=["encoder_queue","encoder_momentum_encoder_queue" ], help="base is codebert/graphcoder/unixcoder",required=False)

    
    # debug
    parser.add_argument('--use_best_mrr_model', action='store_true', help='cosine_space', required=False)
    parser.add_argument('--debug', action='store_true', help='debug mode', required=False)
    parser.add_argument('--n_debug_samples', type=int, default=100, required=False)
    parser.add_argument("--max_codeblock_num", default=10, type=int,
                        help="Optional NL input sequence length after tokenization.")    
    parser.add_argument('--hidden_size', type=int, default=768, required=False)
    parser.add_argument("--eval_frequency", default=1, type=int, required=False)
    parser.add_argument("--mlm_probability", default=0.1, type=float, required=False)

    # model type
    parser.add_argument('--do_avg', action='store_true', help='avrage hidden status', required=False)
    parser.add_argument('--model_type', default="base", choices=["base"], help="ReFCode bi-encoder model type", required=False)
    # moco
    # moco specific configs:
    parser.add_argument('--moco_dim', default=768, type=int,
                        help='feature dimension (default: 768)')
    parser.add_argument('--moco_k', default=32, type=int,
                        help='queue size; number of negative keys (default: 65536), which is divided by 32, etc.')
    parser.add_argument('--moco_m', default=0.999, type=float,
                        help='moco momentum of updating key encoder (default: 0.999)')
    parser.add_argument('--moco_t', default=0.07, type=float,
                        help='softmax temperature (default: 0.07)')

    # options for moco v2
    parser.add_argument('--mlp', action='store_true',help='use mlp head')

    ## Required parameters
    parser.add_argument("--train_data_file", default="dataset/java/train.jsonl", type=str, required=False,
                        help="The input training data file (a json file).")
    parser.add_argument("--output_dir", default="saved_models/pre-train", type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--eval_data_file", default="dataset/java/valid.jsonl", type=str,
                        help="An optional input evaluation data file to evaluate the MRR(a jsonl file).")
    parser.add_argument("--test_data_file", default="dataset/java/test.jsonl", type=str,
                        help="An optional input test data file to test the MRR(a josnl file).")
    parser.add_argument("--codebase_file", default="dataset/java/codebase.jsonl", type=str,
                        help="An optional input test data file to codebase (a jsonl file).")  
    
    parser.add_argument("--lang", default="java", type=str,
                        help="language.")  
    
    parser.add_argument("--model_name_or_path", default="DeepSoftwareAnalytics/CoCoSoDa", type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="DeepSoftwareAnalytics/CoCoSoDa", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="DeepSoftwareAnalytics/CoCoSoDa", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    
    parser.add_argument("--nl_length", default=50, type=int,
                        help="Optional NL input sequence length after tokenization.")    
    parser.add_argument("--code_length", default=100, type=int,
                        help="Optional Code input sequence length after tokenization.") 
    parser.add_argument("--data_flow_length", default=0, type=int,
                        help="Optional Data Flow input sequence length after tokenization.",required=False) 
    
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the test set.")  
    
    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size for evaluation.")
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=4, type=int,
                        help="Total number of training epochs to perform.")

    parser.add_argument('--seed', type=int, default=3407,
                        help="random seed for initialization")  
        

    # ReFCode uncertainty-aware contrastive learning + BM25 hard negatives
    parser.add_argument('--use_refcode_uncertainty', action='store_true', help='Enable ReFCode uncertainty-aware contrastive training')
    parser.add_argument('--refcode_temperature', type=float, default=0.03, help='temperature tau for ReFCode losses')
    parser.add_argument('--refcode_uncertainty_samples', type=int, default=4, help='number of Gaussian samples; choose sample closest to deterministic z')
    parser.add_argument('--refcode_intra_weight', type=float, default=1.0, help='weight for intra-modal uncertainty contrastive loss')
    parser.add_argument('--refcode_kl_weight', type=float, default=1e-5, help='weight for Gaussian KL regularization')
    parser.add_argument('--refcode_symmetric_inter', action='store_true', help='also train Code->NL inter-modal loss')
    parser.add_argument('--use_refcode_global_hard_negative', action='store_true', help='use query-query cosine + BM25 k/10 hard negative mining')
    parser.add_argument('--refcode_hard_idx_file', type=str, default='', help='offline global TopK+BM25 hard negative index pkl; enables paper-style global hard codes')
    parser.add_argument('--refcode_hard_weight', type=float, default=1.0, help='extra denominator weight for selected BM25 hard negative; 1.0 matches the paper denominator')
    parser.add_argument('--refcode_hard_mode', type=str, default='batch_all', choices=['single', 'batch_all'], help='single: previous [B,B+1] hard loss; batch_all: ReFCode-like [B,2B] denominator with B*B hard logits')
    parser.add_argument('--refcode_no_mask_self_in_hard_batch', action='store_true', help='do not mask true-positive codes if they appear in the hard-code batch; normally keep this off')
    parser.add_argument('--refcode_hn_bm25_rank_ratio', type=float, default=0.1, help='choose this BM25 rank ratio from cosine top candidates; 0.1 ~= k/10')
    parser.add_argument('--refcode_hn_candidate_topk', type=int, default=-1, help='top-k query-query cosine candidates; -1 means whole batch minus self')
    parser.add_argument('--use_refcode_augmented_views', action='store_true', help='use optional LLM-generated semantically similar NL/code fields')
    parser.add_argument('--refcode_aug_nl_weight', type=float, default=0.05, help='weight for LLM-generated NL positive view')
    parser.add_argument('--refcode_aug_code_weight', type=float, default=0.02, help='weight for LLM-generated code positive view; keep small')

    # CTRD: Code-Text Relevance Detection auxiliary task.
    # Aggressive default settings are designed to maximize gains on Java first.
    parser.add_argument('--use_ctrd', action='store_true', help='enable CTRD relevance auxiliary loss on top of ReFCode')
    parser.add_argument('--ctrd_weight', type=float, default=0.25, help='overall CTRD loss weight; aggressive default')
    parser.add_argument('--ctrd_batch_topk', type=int, default=16, help='top-K in-batch hard codes per query for CTRD negatives')
    parser.add_argument('--ctrd_hard_weight', type=float, default=2.0, help='BCE weight for offline ReFCode hard negative pairs')
    parser.add_argument('--ctrd_batch_weight', type=float, default=1.0, help='BCE weight for dynamic in-batch hard negative pairs')
    parser.add_argument('--ctrd_rank_weight', type=float, default=0.5, help='extra pairwise ranking loss weight inside CTRD')
    parser.add_argument('--ctrd_rank_margin', type=float, default=0.2, help='margin in softplus(neg - pos + margin) for CTRD ranking')
    parser.add_argument('--ctrd_use_offline_hard', action='store_true', help='include offline ReFCode global hard negatives in CTRD')
    parser.add_argument('--ctrd_detach_hard_code', action='store_true', help='detach offline hard code vector inside CTRD to save/steady gradients')

    # Self-mined hard negatives: mine true model failures using a previous best checkpoint,
    # then train a listwise loss that directly optimizes gold-vs-failure cosine ranking.
    parser.add_argument('--use_self_mined_hard_negative', action='store_true', help='enable listwise loss on self-mined hard negatives')
    parser.add_argument('--self_mined_idx_file', type=str, default='', help='pickle from refcode/failure_harvesting/run.py; contains self_mined_idx list-of-lists')
    parser.add_argument('--self_mined_weight', type=float, default=0.35, help='overall weight for self-mined listwise hard-negative loss')
    parser.add_argument('--self_mined_topk', type=int, default=16, help='keep top-K mined negatives per query in dataset')
    parser.add_argument('--self_mined_train_k', type=int, default=4, help='randomly use this many mined negatives per batch step; reduce if OOM')
    parser.add_argument('--self_mined_temperature', type=float, default=0.03, help='temperature for self-mined listwise CE')
    parser.add_argument('--self_mined_margin', type=float, default=0.02, help='margin for hardest-mined negative softplus term')
    parser.add_argument('--self_mined_max_weight', type=float, default=0.2, help='extra weight for hardest-mined negative term')
    parser.add_argument('--self_mined_false_negative_margin', type=float, default=999.0, help='mask mined negatives with sim > pos_sim + margin; 999 disables')

    # Lightweight training-stage Late Interaction branch.
    # Main/global branch still uses full BATCH_SIZE=128, so in-batch negatives are unchanged.
    parser.add_argument('--use_lite_late_interaction_train', type=int, default=0,
                        help='enable lightweight training-stage late interaction loss')
    parser.add_argument('--li_weight', type=float, default=0.05,
                        help='weight for lightweight late interaction training loss')
    parser.add_argument('--li_temperature', type=float, default=0.05,
                        help='temperature for late interaction CE loss')
    parser.add_argument('--li_train_k', type=int, default=1,
                        help='number of self-mined negatives used by late interaction training')
    parser.add_argument('--li_sample_size', type=int, default=16,
                        help='number of examples sampled from each full batch for LI auxiliary branch')
    parser.add_argument('--li_every_n_steps', type=int, default=2,
                        help='compute LI auxiliary branch every N steps')
    parser.add_argument('--li_chunk_size', type=int, default=16,
                        help='chunk size for LI auxiliary branch')
    parser.add_argument('--li_nl_length', type=int, default=64,
                        help='max NL token length used only by LI auxiliary branch')
    parser.add_argument('--li_code_length', type=int, default=128,
                        help='max code token length used only by LI auxiliary branch')

    # Use fixed valid-set fusion MRR to select checkpoint.
    # This makes training/validation/test consistent with the final global-local fusion inference.
    parser.add_argument('--use_valid_fusion_select', type=int, default=0,
                        help='select checkpoint by validation fusion MRR instead of global bi-encoder MRR')
    parser.add_argument('--valid_fusion_topk', type=int, default=50,
                        help='topK candidates reranked by late interaction on validation')
    parser.add_argument('--valid_fusion_alpha', type=float, default=0.5,
                        help='fixed fusion alpha for validation checkpoint selection')
    parser.add_argument('--valid_rerank_batch_size', type=int, default=64,
                        help='batch size for validation late-interaction reranking')
    parser.add_argument('--valid_fusion_fp16', type=int, default=1,
                        help='use fp16 autocast for validation late-interaction reranking')

    #print arguments
    args = parser.parse_args()
    return  args                     

def create_model(args,model,tokenizer, config=None):
    # logger.info("args.data_aug_type %s"%args.data_aug_type)
    # replace token with type
    if args.data_aug_type in ["replace_type" , "other"] and not args.only_save_the_nl_code_vec:
        special_tokens_dict = {'additional_special_tokens': all_special_token}
        logger.info(" new token %s"%(str(special_tokens_dict)))
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        model.resize_token_embeddings(len(tokenizer))
  
    if (args.loaded_model_filename) and ("pytorch_model.bin" in args.loaded_model_filename):
        logger.info("reload pytorch model from {}".format(args.loaded_model_filename))
        model.load_state_dict(torch.load(args.loaded_model_filename),strict=False) 
        # model.from_pretrain
    model = Model(model)
    if (args.loaded_model_filename) and ("pytorch_model.bin" not in args.loaded_model_filename) :
        logger.info("reload model from {}".format(args.loaded_model_filename))
        model.load_state_dict(torch.load(args.loaded_model_filename), strict=not (getattr(args, 'use_refcode_uncertainty', False) or getattr(args, 'use_ctrd', False))) 
        # strict=False lets ReFCode heads be initialized when loading an older bi-encoder checkpoint.
        # model.from_pretrained(args.loaded_model_filename)  
    if (args.loaded_codebert_model_filename) :
        logger.info("reload pytorch model from {}".format(args.loaded_codebert_model_filename))
        model.load_state_dict(torch.load(args.loaded_codebert_model_filename),strict=False)   
    logger.info(model.model_parameters())


    return model

def main():
    
    args = parse_args()
    #set log
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO )
    #set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    logger.info("device: %s, n_gpu: %s",device, args.n_gpu)
    
    pool = multiprocessing.Pool(cpu_cont)

    # Set seed
    set_seed(args.seed)

    #build model

    if "codet5" in   args.model_name_or_path:
        config = T5Config.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
        tokenizer =  RobertaTokenizer.from_pretrained(args.tokenizer_name)
        model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
        model = model.encoder
    else:
        config = RobertaConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
        tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
        model = RobertaModel.from_pretrained(args.model_name_or_path) 
    model=create_model(args,model,tokenizer,config)

    logger.info("Training/evaluation parameters %s", args)
    args.start_step = 0

    model.to(args.device)
    
    # Training
    if args.do_multi_lang_continue_pre_train:
        multi_lang_continue_pre_train(args, model, tokenizer, pool)
        output_tokenizer_dir = os.path.join(args.output_dir,"tokenzier")                      
        if not os.path.exists(output_tokenizer_dir):
            os.makedirs( output_tokenizer_dir)    
        tokenizer.save_pretrained( output_tokenizer_dir)
    if args.do_train:
        train(args, model, tokenizer, pool)
     
    
    # Evaluation
    results = {}

    if args.do_eval:
        checkpoint_prefix = 'checkpoint-best-mrr/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        if (not args.only_save_the_nl_code_vec) and (not args.do_zero_short) :
            model.load_state_dict(torch.load(output_dir),strict=False)      
        model.to(args.device)
        result=evaluate(args, model, tokenizer,args.eval_data_file, pool)
        logger.info("***** Eval valid results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key],4)))
            
    if args.do_test:

        logger.info("runnning test")
        checkpoint_prefix = 'checkpoint-best-mrr/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        if (not args.only_save_the_nl_code_vec) and (not args.do_zero_short) :
            model.load_state_dict(torch.load(output_dir),strict=False)      
        model.to(args.device)
        result=evaluate(args, model, tokenizer,args.test_data_file, pool)
        logger.info("***** Eval test results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key],4)))
        save_json_data(args.output_dir, "result.jsonl", result)
    return results


if __name__ == "__main__":
    main()
