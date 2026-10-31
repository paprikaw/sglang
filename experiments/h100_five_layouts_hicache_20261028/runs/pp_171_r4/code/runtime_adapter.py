"""Private fair small-Expert policy, native Attention and communication."""
def install():
    import h100_runtime_adapter, diagnostic_switch, fragmentation, ep_fragmentation
    h100_runtime_adapter.install()
    diagnostic_switch.install()
    fragmentation.install()
    ep_fragmentation.install()

_base_install=install
def install():
    _base_install()
    import hicache_audit
    hicache_audit.install()

_hicache_install=install
def install():
    _hicache_install()
    import pp_optim
    pp_optim.install()
