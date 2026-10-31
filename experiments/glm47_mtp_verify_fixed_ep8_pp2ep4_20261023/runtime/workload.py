"""Frozen semantic prompts; no random token IDs and no synthetic KV."""
from array import array
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import sys

INPUTS = Path(__file__).resolve().parents[1] / 'semantic_inputs' / 'p4096'

def input_dir():
    return Path(os.environ.get('GLM_SEMANTIC_INPUTS',str(INPUTS)))


@lru_cache(maxsize=1)
def manifest():
    value = json.loads((input_dir() / 'manifest.json').read_text())
    if value['schema'] != 'glm47-semantic-inputs-v1' or not value['kv_requires_actual_model_prefill']:
        raise ValueError('Semantic context provenance required')
    return value


@lru_cache(maxsize=1)
def prompts():
    rows = []
    for row in manifest()['requests']:
        data = (input_dir() / row['file']).read_bytes()
        if hashlib.sha256(data).hexdigest() != row['sha256']:
            raise ValueError('Semantic prompt digest changed')
        ids = array('I')
        ids.frombytes(data)
        if sys.byteorder != 'little':
            ids.byteswap()
        if len(ids) != row['tokens']:
            raise ValueError('Semantic prompt length changed')
        rows.append(list(ids))
    if len(rows) != 8 or len({tuple(ids[:64]) for ids in rows}) != 8:
        raise ValueError('Eight distinct semantic contexts required')
    return rows

def prefixes(length, count=8):
    if length != manifest()['raw_prefix_tokens'] or count != 8:
        raise ValueError('Only the frozen semantic cohort is admitted')
    return [ids[:length] for ids in prompts()]

def query(index, count, warm):
    if count != 1 or not 0 <= index < 8:
        raise ValueError('Semantic query must use the actual final prompt token')
    return prompts()[index][manifest()['raw_prefix_tokens']:]

def near_capacity(capacity, batch):
    return ((int(capacity*.90)//batch-384)//64)*64

def cached_prefix_length(raw_prefix_length):
    # Native EAGLE keys span adjacent raw tokens. The request matcher reserves
    # its final input token before bigram conversion and rounds to full pages.
    return ((raw_prefix_length - 1) // 64) * 64
