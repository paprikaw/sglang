"""Save collected requests only on failure; never promote a partial episode."""


def replay_with_failure_capture(replay_fn, save, partial_path, *args, **kwargs):
    rows = []
    try:
        return replay_fn(*args, partial_rows=rows, **kwargs)
    except BaseException as exc:
        try:
            save(partial_path, dict(passed=False, complete=False,
                error_type=type(exc).__name__, error=str(exc),
                collected_requests=len(rows), partial_requests=rows,
                scope='Failure diagnosis only; not a complete episode or a performance mean'))
        except Exception as capture_error:
            exc.add_note(f'Partial diagnostic write failed: {capture_error!r}')
        raise
