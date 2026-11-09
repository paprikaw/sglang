# Private HiCache run cancellation and time admission

2026-11-09 最新范围：同暖512边界的 sync/cpu_spill GPU 往返及本次组合742项CPU检查已通过，具体源码、证据和未完成项见[基线验收摘要](BASELINE_ACCEPTANCE_20261109.md)。下文保留各历史检查点及失败记录；历史“尚未验证”的表述仅适用于对应版本，不代表当前扩大了冷暖生成、其他策略、Graph 或GPU归还的资格。

`hicache_guard_run.py --run RUN` starts the unchanged canonical `dashboard_guard.py run`. Its budget begins before Guard is spawned. At the work cutoff it writes a run-specific cancellation request; it does not signal Guard. The controller keeps SSH and Guard alive while the worker's independent cleanup thread terminates only the recorded children. Child launch and registration share a lock with cleanup. Cancellation before worker startup prevents a later model launch.

The acknowledgement binds the frozen run nonce, authenticated lease ID, worker PID/start ticks/host/boot identity, cleanup interval and complete owned-child list. All four cleanup checks must come from the fresh cleanup result. SSH termination, a finished request, an old cleanup file and a usable source pool are insufficient. A successful experiment also requires its original exact output/state assertions, complete hashed diagnostic archive and successful canonical Guard release. Experiment failure with verified cleanup can still release the lease normally.

The candidate limits, measured from the supervisor's earlier origin, are work 250 s, remote cleanup 285 s, Guard stop 290 s and total 300 s. The original cleanup helper runs in a single daemon thread; retries share the same absolute deadline. An independent timer exits the worker at 285 s even if cleanup or its cancellation-file reader is blocked. The controller waits until that deadline rather than treating SSH exit as remote exit. If Guard is still alive at 290 s, the supervisor authenticates this run's receipt and records `release(False)` before terminating its own Guard process (TERM wait 3 s, then KILL wait 3 s). A settlement error remains an error. Missing or false acknowledgement cannot become a verified cleanup.

The registry's existing quarantine rules remain in force. The default grace is 90 s, after which the original registry may record `released_unverified`; quarantine is not permanent and this state is not a safe release. An unexpected already-expired lease also remains failed. If Guard has actually exited while its lease is still active, the adapter can use the original heartbeat implementation while awaiting the same bounded acknowledgement and settling the authenticated lease. It does not change the canonical heartbeat interval or global Guard behavior.

The r13 first reference arrived about 163.5 s after Guard start. The four reference service times sum to 38.5666 s; 29.4031 s between them was outside request service and was not attributed to one operation. The pre-worker interval was about 61.14 s, including a 46.92 s lease-to-binding-file window that covers admission work as well as writing. The r13 ready log is about +154.5 s at one-second precision; this is not a 155 s conservative startup bound. Corresponding ready logs in r11/r12 were about +101.2/+75.2 s. Full static qualification took 0.0776 s in a separate CPU check, so it remains intact in both prepare and worker.

The only critical-path I/O change is to persist full pre-serialization request contracts and stage diagnostics in a private, fsynced node-local directory, then copy and content-hash-check them after model cleanup. Actual worker CPU tests compare both paths and count the removed synchronous shared writes. This proves removal of those writes, not a measured 29.4 s speedup. Phase timestamps, full qualification and per-control authentication timings remain recorded. Reference inputs, sampling settings, output lengths and all live transition/loadback checks are unchanged.

Time admission requires 105 s of work budget before the four references, 60 s before grow and 32 s before shrink. Thus reference admission requires actual readiness by approximately supervisor +145 s; r13-like late readiness is rejected cleanly. The 105 s allowance consists of 42 s for four references (observed maximum sum 38.57 s), 3 s for initial demotion/control and 60 s remaining. The 60 s allows two 20 s active requests (r11 observed 19.57/17.87 s including overlap fences), an additional 5 s per transition for unmeasured spill cost, 7 s for control/state checks and 3 s for final prefix restoration. The 32 s second-stage allowance is 20+5+4+3 s. The extra spill allowance and control margins are planning assumptions, not GPU measurements. A later admission failure or work cutoff cancels instead of shortening requests or borrowing cleanup time.

The 35 s work-to-cleanup window is an absolute success/failure boundary, not a promise that a persistently failing GPU cleanup will succeed. CPU tests use real processes, the original temporary registry and 15 s TTL heartbeat, delayed cleanup across TTL, late SSH exit, hung cleanup/observer and quarantine before Guard termination. Their GPU-idle fields are explicitly CPU stand-ins. GPU spill release, restore and continuity still require a fresh Guard run. The r10 cold/warm failure, r12 alias failure and r13 pre-migration failure remain unchanged; cold/warm generation qualification remains false.

The initial ff9a2d64 delivery is backed by 91 related CPU checks (0 failures/errors/skips; pytest 52.805 s). Their source hashes match before and after, and the canonical Guard, registry and cache adapter hashes are unchanged. The actual worker comparison removes all four shared request-contract writes and shared stage writes from the serving critical path, preserving their complete content in the verified post-cleanup archive. Cancellation/progress write failures are also injected: they leave the controller waiting for the fresh acknowledgement rather than abandoning the remote worker. Results and command metadata are stored in the shared `cpu-task-lifecycle-20261109` artifact directory. This check replaces no historical GPU result and proves no CUDA cleanup or spill restoration.

Independent review of that checkpoint found three private verdict defects: final task progress was newer than cleanup, final acknowledgement reads were excluded from elapsed time, and a cancelled completion could retain a true raw verdict. Three focused red cases reproduce those failures. The follow-up writes final progress before cleanup, includes final verification in total elapsed time, and clears the worker verdict on every exception. The same test bytes then pass in 41 affected controller, supervisor and actual-worker checks (0 failures/errors/skips; pytest 58.957 s), with unchanged source hashes throughout and unchanged canonical Guard files. Synchronous shared-filesystem and registry calls themselves are not proven to return by a deadline; a delayed final read must fail the reported total budget even after verified process cleanup.
# Minimal overhead observation after r14

`hicache_transition_smoke.py --prepare --overhead-sample` retains the reviewed
sync/native `cpu_spill`, warm512 reference, exact IDs and state/cleanup gates.
It adds one 513-input/256-output no-transition request before each direction.
Grow baseline and active request both load 512 tokens from CPU; a second
demotion after the baseline preserves that condition. Shrink baseline and
active request both hit 512 device tokens in the PP3 source layout.

Each direction reports one client E2E/TTFT/mean-TPOT observation and a signed
transition-minus-baseline difference in milliseconds. Existing per-rank
coordinator preparation, fence and total-through-cleanup times are retained.
These samples establish neither a latency distribution nor an isolated causal
cost. Token interval pause, isolated transfer/restore, and strict peak memory
remain unmeasured. The original cold/warm-generation limitation remains.

The 250/285/290/300-second work/remote-cleanup/Guard/total limits are unchanged.
Admission reserves 150 seconds before the four original references, 105/52
before the extra grow/shrink reference, and 80/32 before the transitions.
R14 observed 163.294 seconds including cleanup; two additional full requests
are planned at about 40 seconds together. This is an estimate, so insufficient
remaining time cancels this run rather than shortening requests or cleanup.
