"""Normal checkpoint loading from a topology-independent canonical RAM bank."""

import copy
import logging
import os
import time

try:
    from .cpu_canonical import model_view
except ImportError:
    from cpu_canonical import model_view  # CPU-only direct-module tests.

logger = logging.getLogger(__name__)


class CanonicalCpuModelLoader:
    def __init__(self, load_config, fallback):
        self.load_config = load_config
        self.fallback = fallback

    def download_model(self, model_config):
        # Download cannot populate a bank implicitly or silently hit disk.
        with model_view(
            model_config.model_path,
            os.environ["SGLANG_CANONICAL_WEIGHT_CACHE_SOCKET"],
            os.environ.get("SGLANG_CANONICAL_WEIGHT_CACHE_SOURCE_COMMIT", ""),
        ):
            pass

    def load_model(self, *, model_config, device_config):
        from sglang.srt.runtime_context import get_model

        flags = get_model()
        if flags.weight_loader_disable_mmap:
            raise ValueError("canonical CPU bank requires mmap checkpoint loading")
        # The ordinary iterator discards its background prefetch handle. Such
        # a thread could outlive the view/pin, so suppress redundant RAM-bank
        # prefetch while preserving its default single-thread loading policy.
        original_extra = self.load_config.model_loader_extra_config
        scoped_extra = dict(original_extra)
        if flags.weight_loader_prefetch_checkpoints and not (
            {"enable_multithread_load", "num_threads"} & scoped_extra.keys()
        ):
            scoped_extra["enable_multithread_load"] = False
        started = time.monotonic()
        with model_view(
            model_config.model_path,
            os.environ["SGLANG_CANONICAL_WEIGHT_CACHE_SOCKET"],
            os.environ.get("SGLANG_CANONICAL_WEIGHT_CACHE_SOURCE_COMMIT", ""),
        ) as (view, bank):
            view_config = copy.copy(model_config)
            view_config.model_path = view
            self.load_config._canonical_weight_cache_view = view
            self.load_config.model_loader_extra_config = scoped_extra
            try:
                with flags.override(weight_loader_prefetch_checkpoints=False):
                    model = self.fallback.load_model(
                        model_config=view_config, device_config=device_config
                    )
            finally:
                # A model may retain this shallow config object. Never leave a
                # deleted temporary view as its persistent model path.
                view_config.model_path = model_config.model_path
                self.load_config.model_loader_extra_config = original_extra
                del self.load_config._canonical_weight_cache_view
            model._canonical_cpu_weight_cache_key = bank["group_key"]
        logger.info(
            "CPU weight cache load: source=canonical_cpu_daemon key=%s elapsed=%.3fs",
            bank["group_key"],
            time.monotonic() - started,
        )
        return model
