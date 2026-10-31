# macOS GPU dashboard command

Install **in the Mac's local terminal**, using the same network and SSH access
normally used for Spartan. No sudo or Python is required on the Mac.

```bash
(set -o pipefail; ssh bxb1@spartan-login2.hpc.unimelb.edu.au \
  'cat /data/gpfs/projects/punim2715/vllm_workbench/engines/sglang/autoscaling/tools/gpu_dashboard_client/install-macos.sh' | /bin/bash) &&
export PATH="$HOME/.local/bin:$PATH" && gpu-dashboard
```

Subsequent terminals can use `gpu-dashboard` directly. The installer adds a
guarded PATH entry to `~/.zprofile` and `~/.zshrc` for macOS's default zsh and
backs up an existing different command. It refuses to install on Linux, so
accidentally running the command in a Spartan terminal does not edit profiles.

The command opens http://127.0.0.1:19510/ in the default macOS browser. It checks
the exact dashboard health marker before reusing a local listener, otherwise
opens an SSH loopback tunnel to spartan-login2:19510. The tunnel is independent
of the terminal after SSH authentication. It can expire after connection loss;
running the command again establishes a new connection as needed.

If a local SSH alias is needed, use
`GPU_DASHBOARD_SSH_HOST=your-spartan-alias gpu-dashboard`. A different local port
can be selected with `GPU_DASHBOARD_PORT=19511 gpu-dashboard`. These overrides
do not change the remote service's fixed port. `--no-open` validates the tunnel
and prints the URL without launching a browser.

`gpu-dashboard` is the launcher source. Run `python3 build_installer.py` in this
directory after editing it to regenerate the self-contained installer.
