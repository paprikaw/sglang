"""Private per-process SM80 Humming cache with the existing A100 seed."""
import os
from pathlib import Path
import shutil
from common import SEED

def isolate_humming_cache():
    root=Path('/tmp')/f'{os.getuid()}-{Path(os.environ["RUN_DIR"]).name}'/os.environ['ARM']/str(os.getpid())
    for key,name in [('HUMMING_CACHE_DIR','cache'),('HUMMING_TMP_DIR','tmp')]:
        path=root/name
        if not path.exists() and name=='cache' and SEED.is_dir():
            assert not any(p.is_symlink() for p in SEED.rglob('*'))
            assert sum(p.stat().st_size for p in SEED.rglob('*') if p.is_file()) <= 32*1024**2
            shutil.copytree(SEED,path,ignore=shutil.ignore_patterns('lock','*.lock'))
        path.mkdir(parents=True,exist_ok=True);os.environ[key]=str(path)
    os.environ['HUMMING_DISABLE_PARALLEL_BUILD']='1'
    return root
