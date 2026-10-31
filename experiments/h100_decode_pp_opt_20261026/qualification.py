"""Separate correctness diagnostic; none of these cohorts is a speed sample."""
import hashlib
import os
from pathlib import Path
import time


def install(base):
    import hopper_reference
    original_env = base.environment

    def environment(cfg, arm, inherited):
        result = original_env(cfg, arm, inherited)
        result.update(PP_READY_POLICY_PATH=cfg['policy_path'],
            PP_DECODE_OPT_POLICY_PATH=cfg['decode_opt_policy_path'],
            PP_TRANSFER_AUDIT_CONTROL=cfg['transfer_audit_control'])
        return result

    base.environment = environment
    base.engine_args = hopper_reference.engine_args

    def run_workloads(run, cfg, arm, key, health):
        import json
        import pp_ready_send
        from pp_transfer_audit import compare_transfers
        from transformers import AutoTokenizer
        if arm != 'pp2':
            raise RuntimeError('This diagnostic is PP only')
        tokenizer = AutoTokenizer.from_pretrained(cfg['model_path'], local_files_only=True)
        tokens = []
        for index in range(8):
            head = [5000+index]+tokenizer.encode(f'Document {index}. ', add_special_tokens=False)
            body = tokenizer.encode('A distributed service keeps a record of independent requests. '
                'Explain the computation and communication clearly.\n', add_special_tokens=False)
            tokens.append((head+body*(4096//len(body)+2))[:4096])
        pref = [base.packet(cfg, arm, t, 1, i) for i,t in enumerate(tokens)]
        decode = [base.packet(cfg, arm, t+[198], 128, i) for i,t in enumerate(tokens)]
        base.h.save(run/arm/'prompt_fingerprints.json', dict(prefix_tokens=4096,
            sha256=[hashlib.sha256(str(t).encode()).hexdigest() for t in tokens]))

        def control(value):
            # Preserve the mapped inode; no replacement or truncate.
            fd = os.open(cfg['transfer_audit_control'], os.O_WRONLY | os.O_NOFOLLOW)
            try:
                if os.write(fd, value.encode()) != 1:
                    raise RuntimeError('Capture control write failed')
            finally:
                os.close(fd)

        def policy(early):
            pp_ready_send.set_policy(cfg['decode_opt_policy_path'], early)
            pp_ready_send.set_policy(cfg['policy_path'], False)

        def cached(result):
            if len(result['requests']) != 8 or any(
                r['cached_tokens'] != 4096 or r['output_tokens'] != 128
                for r in result['requests']):
                raise RuntimeError('Diagnostic workload/cache differs')

        control('0');policy(False);base.flush_cache(cfg, key)
        base.h.save(run/arm/'prefix_preparation.json', dict(real_model_prefix=True,
            measured=False, result=base.cohort(cfg,key,pref)))
        for count in (2,4,6,8,8):
            health();result=base.cohort(cfg,key,decode[:count])
            if any(r['cached_tokens'] != 4096 for r in result['requests']):
                raise RuntimeError('Warmup prefix mismatch')
        audits=[]
        for index, early in enumerate((False,True,True,False,False,True), 1):
            health();policy(early);cached(base.cohort(cfg,key,decode))
            control(str(index))
            base.progress(run,cfg,f'pp2_capture_{index}',performance_sample=False)
            result=base.cohort(cfg,key,decode);cached(result)
            # Normal control requests traverse both stages. Retire final output
            # feedback under the original protocol before ending capture.
            for _ in range(3):
                base.h.http_json(cfg['port'],'/get_server_info',key,timeout=30)
            control('0')
            paths=[run/arm/'transfer_audit'/f'epoch{index}.pp{rank}.json' for rank in (0,1)]
            deadline=time.monotonic()+45
            while not all(p.is_file() for p in paths):
                health()
                base.h.http_json(cfg['port'],'/get_server_info',key,timeout=30)
                if time.monotonic()>deadline:
                    raise TimeoutError('Both idle-stage capture receipts were not published')
                time.sleep(.05)
            receipts=[json.loads(p.read_text()) for p in paths]
            audit=compare_transfers(receipts)
            row=dict(epoch=index,mode='early_activation' if early else 'baseline',
                performance_sample=False,cohort=result,audit=audit)
            base.h.save(run/arm/f'qualification.epoch{index}.json',row)
            audits.append(dict(epoch=index,mode=row['mode'],**audit))
            if not audit['passed']:
                raise RuntimeError('PP payloads did not match sender to consumer')
        policy(False);control('0')
        summary=dict(passed=True,performance_sample=False,cohorts=6,audits=audits,
            scope='Exact transmitted activation and output bytes, ordering and observed microbatch membership; '
                  'does not assert bitwise autoregressive repeatability or model quality')
        base.h.save(run/arm/'metrics.qualification.json',summary)
        base.h.save(run/arm/'completed.json',summary)

    base.run_workloads=run_workloads
