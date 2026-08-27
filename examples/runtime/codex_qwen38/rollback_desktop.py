"""Remove only this deployment's two Codex routing overrides."""
import os
import tempfile
import tomllib
from pathlib import Path

CONFIG = Path('/home/bxb1/.codex/config.toml')
EXPECTED = {
    'openai_base_url': 'http://127.0.0.1:31038/v1',
    'model_catalog_json': '/data/gpfs/projects/punim2715/model_serving/qwen38-27b-uncensored/codex-models-combined.json',
}


def main():
    before = CONFIG.read_bytes()
    parsed = tomllib.loads(before.decode())
    for key, expected in EXPECTED.items():
        if key in parsed and parsed[key] != expected:
            raise SystemExit(f'{key} was changed by another configuration; refusing to remove it')
    lines = before.decode().splitlines(keepends=True)
    kept = [line for line in lines if not any(line.startswith(key + ' =') for key in EXPECTED)
            and not line.startswith('# Local model routing;')
            and not line.startswith('# Rollback backup: config.toml.before-qwen38-')]
    after = ''.join(kept)
    actual = tomllib.loads(after)
    assert actual == {key: value for key, value in parsed.items() if key not in EXPECTED}
    fd, temporary = tempfile.mkstemp(prefix='.qwen-rollback-', dir=CONFIG.parent)
    try:
        os.fchmod(fd, CONFIG.stat().st_mode & 0o777)
        with os.fdopen(fd, 'w') as stream:
            stream.write(after)
        if CONFIG.read_bytes() != before:
            raise SystemExit('Config changed concurrently; retry rollback')
        os.replace(temporary, CONFIG)
    finally:
        Path(temporary).unlink(missing_ok=True)
    print('Removed Qwen routing/catalog overrides; reconnect the Spartan host in Codex.')


if __name__ == '__main__':
    main()
