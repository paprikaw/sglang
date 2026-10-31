import os
if (os.environ.get('HY3_PROCESS_ROLE')=='server' and os.environ.get('HY3_STREAMING_ACTIVE')=='1'
        and os.environ.get('EXPERIMENT_GUARD_MODEL_CACHE_ADMISSION_ID')):
    try:
        from loader_patch import install
        install()
        from communication_reserve import install as reserve_communication
        reserve_communication()
        from scheduler_hooks import install as install_scheduler
        install_scheduler()
    except BaseException:
        import traceback
        traceback.print_exc()
        os._exit(91)
