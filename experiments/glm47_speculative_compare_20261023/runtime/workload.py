"""Disjoint reproducible prefixes; Q excludes the cached P tokens."""
import random
import hashlib
import struct

def prefixes(length, count=8):
    if length<=0 or length%64 or length+384>131072 or count!=8:
        raise ValueError('Eight page aligned, in-range prefixes required')
    return [[1000+i,*random.Random(20261022+i).choices(range(100,120000),k=length-1)]
            for i in range(count)]

def query(index, count, warm):
    return [120100+index+(32 if warm else 0)] + [120200+index]*(count-1)

def near_capacity(capacity, batch):
    return ((int(capacity*.90)//batch-384)//64)*64

def page_hashes(tokens):
    prior=b'';out=[]
    for i in range(0,len(tokens),64):
        prior=hashlib.sha256(prior+struct.pack('<64I',*tokens[i:i+64])).digest()
        out.append(prior.hex())
    return out


def cached_prefix_length(raw_prefix_length):
    # Native EAGLE keys span adjacent raw tokens. The request matcher reserves
    # its final input token before bigram conversion and rounds to full pages.
    return ((raw_prefix_length - 1) // 64) * 64
