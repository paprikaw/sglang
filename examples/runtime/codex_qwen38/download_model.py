"""Download the explicitly authorized, pinned model snapshot."""
import json
import os
from pathlib import Path
from datetime import datetime, timezone

os.umask(0o077)
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent
REPO = "orcarouter/Qwen3.8-27B-Uncensored"
REVISION = "9878936be9458522b5aeed0e13476bb8426f57f0"
DESTINATION = "/data/gpfs/projects/punim2715/huggingface/Qwen3.8-27B-Uncensored"

if __name__ == "__main__":
    print(f"Downloading {REPO}@{REVISION} to {DESTINATION}", flush=True)
    path = snapshot_download(REPO, revision=REVISION, local_dir=DESTINATION, max_workers=4)
    files = sorted(Path(path).glob("model-*.safetensors"))
    record = {
        "repo": REPO, "revision": REVISION, "path": path,
        "shards": len(files), "weight_bytes": sum(p.stat().st_size for p in files),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    (ROOT / "model_snapshot.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)
