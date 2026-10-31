"""Freeze native SGLang EP2 overlap ON/OFF, reusing the fair GEMM protocol."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def descriptor(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--shared", type=int, choices=[0, 1], required=True)
    parser.add_argument("--tbo", action="store_true")
    args = parser.parse_args()
    run = args.run.resolve()
    if run.exists():
        raise RuntimeError("Run already exists")
    subprocess.run([sys.executable, str(ROOT.parent / "h100_pp_graph_gap_opt_20261027/prepare.py"),
                    "--run", str(run), "--arms", "ep2", "--policies", "0", "--samples", "5"], check=True, stdout=subprocess.DEVNULL)
    code = run / "code"
    shutil.copy2(ROOT / "native_overlap.py", code / "native_overlap.py")
    runtime = code / "runtime_adapter.py"
    runtime.write_text(runtime.read_text() + '''
_before_native_overlap = install
def install():
    _before_native_overlap()
    import native_overlap
    native_overlap.install_runtime()
''')
    runner = code / "runner.py"
    text = runner.read_text()
    marker = "if __name__ == '__main__':"
    assert text.count(marker) == 1
    runner.write_text(text.replace(marker, "import native_overlap\nnative_overlap.install(sys.modules[__name__])\n\n" + marker))
    cfg = read(run / "config.json")
    cfg.update(native_shared_overlap=bool(args.shared), native_two_batch_overlap=args.tbo,
        port=26421, dist_port=26422,
        protocol="Native Qwen shared-expert/DeepEP overlap ON/OFF and optional TBO flag; unchanged EP2 shared Triton GEMM, B8=cached4096+new1->128; five profiler-off Decode cohorts; separate native profiles.")
    save(run / "config.json", cfg)
    spec = read(run / "dashboard_spec.json")
    frozen = [r for r in spec["frozen_inputs"] if not Path(r["path"]).is_relative_to(code) and r["path"] != str(run / "config.json")]
    frozen += [descriptor(f) for f in sorted(code.glob("*.py"))] + [descriptor(run / "config.json")]
    spec.update(title=f"H100 EP2 native overlap shared={args.shared} TBO={args.tbo}", port=26420, frozen_inputs=frozen)
    spec["tasks"][0]["label"] = "Native EP2 overlap, same GEMM and real GPU prefix"
    save(run / "dashboard_spec.json", spec)
    manifest = read(run / "launch_manifest.json")
    manifest.update(native_shared_overlap=bool(args.shared), native_two_batch_overlap=args.tbo,
        private_adapters=[descriptor(f) for f in sorted(code.glob("*.py"))],
        instrumentation="One-time shared-expert stream receipts during warmup/capture; native MoE math and stream code unchanged")
    save(run / "launch_manifest.json", manifest)
    print(json.dumps(dict(run=str(run), shared=bool(args.shared), tbo=args.tbo)))


if __name__ == "__main__":
    main()
