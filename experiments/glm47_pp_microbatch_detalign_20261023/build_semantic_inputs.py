"""Build semantic token inputs; NEVER constructs KV tensors.

Eight distinct streams of real source code/docstrings from the frozen local
SGLang checkout avoid long-prefix sharing. The actual model must Prefill them.
"""
from array import array
import argparse
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'runtime'))
from common import ENGINE, MODEL


def digest(data):
    return hashlib.sha256(data).hexdigest()


def build(prefix_tokens=4096,engine=ENGINE,commit=None):
    from transformers import AutoTokenizer
    if prefix_tokens not in (4096,16384):
        raise ValueError('This narrow A/B admits only P4096 or P16384')
    if commit is None or len(commit)!=40:
        raise ValueError('Pin the semantic source commit explicitly')
    target = ROOT / 'semantic_inputs' / f'p{prefix_tokens}'
    target.parent.mkdir(exist_ok=True)
    target.mkdir(exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True,
                                             trust_remote_code=True)
    raw_prompt_length = prefix_tokens+1
    files = sorted((engine / 'python/sglang').rglob('*.py'))
    all_ids, provenance = [], []
    suffix_text = ('\n\nThe final code excerpt may be truncated. Based on these reference '
                   'files, explain how an inference server schedules and executes requests. '
                   'Give a detailed answer with at least eight numbered points, using '
                   'complete sentences. Do not reproduce the source files.\n')
    suffix = tokenizer.encode(suffix_text, add_special_tokens=False)
    generation = tokenizer.encode('<|assistant|></think>', add_special_tokens=False)
    for index in range(8):
        header_text = (f'<|user|>Reference set {index + 1}. Read the following source '
                       'code and its documentation, then answer the question at the end.\n')
        header = tokenizer.encode(header_text, add_special_tokens=True)
        needed = raw_prompt_length - len(header) - len(suffix) - len(generation)
        body, used = [], []
        for path in files[index::8]:
            data = path.read_bytes()
            text = data.decode('utf-8').replace('<|', '< |').replace('[gMASK]', '[g MASK]')
            excerpt = f'\n\n### File: {path.relative_to(engine)}\n```python\n{text}\n```\n'
            tokens = tokenizer.encode(excerpt, add_special_tokens=False)
            take = min(len(tokens), needed - len(body))
            body.extend(tokens[:take])
            used.append(dict(path=str(path), source_sha256=digest(data),
                             tokens_consumed=take, file_excerpt_tokens=len(tokens)))
            if len(body) == needed:
                break
        if len(body) != needed:
            raise RuntimeError('Insufficient distinct semantic source material')
        ids = header + body + suffix + generation
        assert len(ids) == raw_prompt_length
        packed = array('I', ids)
        assert packed.itemsize == 4 and sys.byteorder == 'little'
        data = packed.tobytes()
        name = f'request{index}.u32le'
        (target / name).write_bytes(data)
        all_ids.append(ids)
        provenance.append(dict(request_index=index, file=name, sha256=digest(data),
                               tokens=len(ids), source_files=used,
                               decoded_tail=tokenizer.decode(ids[-180:])))
    assert len({tuple(ids[:64]) for ids in all_ids}) == 8
    sanity = []
    for index, question in enumerate((
        'Write a Python function named add_numbers that returns the sum of two integers. '
        'Then explain the function and give three examples.',
        'Explain why the Earth has seasons. Give a clear answer in complete sentences '
        'and explain the role of the tilt of its axis.',
    )):
        ids = tokenizer.apply_chat_template([dict(role='user', content=question)],
                                             tokenize=True, add_generation_prompt=True,
                                             enable_thinking=False)
        if isinstance(ids, Mapping):
            ids = ids['input_ids']
        if ids and isinstance(ids[0], list):
            raise RuntimeError('Sanity prompt must be one unbatched sequence')
        ids = [int(token) for token in ids]
        sanity.append(dict(index=index, text=question, input_ids=ids))
    manifest = dict(schema='glm47-semantic-inputs-v1', source_commit=commit,
                    model_path=str(MODEL), raw_prompt_tokens=raw_prompt_length,
                    raw_prefix_tokens=prefix_tokens, query_tokens=1,
                    expected_native_bigram_cached_tokens=((prefix_tokens-1)//64)*64,
                    request_count=8, requests=provenance, sanity_prompts=sanity,
                    input_ids_are_semantic=True, kv_tensors_constructed=False,
                    kv_requires_actual_model_prefill=True, shared_long_prefix=False,
                    tokenizer_config_sha256=digest((MODEL / 'tokenizer_config.json').read_bytes()))
    (target / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(path=str(target), requests=8, tokens_per_request=raw_prompt_length,
                          corpus_kind='Distinct real SGLang source code and documentation',
                          kv_constructed=False)))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--prefix-tokens',type=int,choices=(4096,16384),default=4096)
    parser.add_argument('--engine-root',type=Path,default=ENGINE)
    parser.add_argument('--engine-commit',required=True)
    args=parser.parse_args()
    import subprocess
    actual=subprocess.check_output(['git','-C',str(args.engine_root),'rev-parse','HEAD'],text=True).strip()
    if actual!=args.engine_commit:raise RuntimeError('Semantic corpus source commit changed')
    build(args.prefix_tokens,args.engine_root,args.engine_commit)
