"""Shared E10 real-weight validation plus matched EP cohort admission."""
def install():
    import a100_runtime_adapter, runtime_setup, prefill_admission
    a100_runtime_adapter.install()
    runtime_setup.install()
    prefill_admission.install()
