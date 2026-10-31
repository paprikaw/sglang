"""Frozen, hardware-specific accepted adapter plus a diagnostic-only selector."""
import os


def install():
    if os.environ['EP_STEADY_HARDWARE'] == 'A100':
        import a100_runtime_adapter
        a100_runtime_adapter.install()
    else:
        import h100_runtime_adapter
        h100_runtime_adapter.install()
        import diagnostic_switch
        diagnostic_switch.install()

_previous_install = install
def install():
    _previous_install()
    import fragmentation
    fragmentation.install()
    import ep_fragmentation
    ep_fragmentation.install()
