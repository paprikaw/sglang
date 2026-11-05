# Research development workflow

For this fork's EP, PP, HiCache and P/D research, read
`research/pp_elastic/README.md`, `VALIDATION.md` and `WORKFLOW.md` in that directory
before changing code. Preserve their distinctions between available code and
validated execution paths.

- The shared baseline is `codex/pp-elastic-integration`. The PP + HiCache project
  integrates on `codex/pp-hicache-autoscaling` through its draft umbrella PR.
- Use one task branch and one worktree per contributor. A worktree alone does
  not isolate a branch. The coordinator owns the project integration branch;
  contributors open PRs targeting it and do not push directly to it.
- Commit and push each coherent checkpoint, before a long experiment, and before
  handing off or ending a work session. During extended implementation, aim for
  a reviewable pushed checkpoint every 30-60 minutes. Label incomplete work and
  keep its PR draft; one PR may contain multiple checkpoint commits.
- Stage explicit task files. Preserve other worktrees, dirty changes, frozen
  experiment tags and independent dependency repositories. Never force-push a
  shared branch or publish credentials, model weights or bulk experiment data.
- Each PR records its parent/dependencies, source and target branches, changes,
  tests actually run and known limits. The coordinator keeps child PR links and
  integration status in the umbrella PR and attaches created PRs to the task.
- Merge reviewed child PRs into the project branch sequentially, then validate
  the combined revision before completing the umbrella PR. Use merge commits
  between these shared branches so their ancestry and dependencies remain clear.
- Pin experiments to a pushed commit and immutable experiment tag, with runtime
  adapters and dependency versions recorded. Follow Experiment Guard before GPU
  use. Historical baseline tests do not validate new dynamic PP + HiCache code.
