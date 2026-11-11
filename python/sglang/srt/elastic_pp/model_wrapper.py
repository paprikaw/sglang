"""Model-facing elastic PP hooks shared by decoder/LM wrappers."""

from __future__ import annotations

from contextlib import contextmanager
from threading import RLock

from sglang.srt.elastic_pp.weight_profile import FirstLoadWeightProfiler


class ElasticPPModelWrapperMixin:
    def _init_elastic_pp_weight_profile(self, enabled: bool) -> None:
        self._elastic_pp_weight_profiler = (
            FirstLoadWeightProfiler(self.config.num_hidden_layers) if enabled else None
        )
        self._elastic_pp_weight_load_lock = RLock()
        self._elastic_pp_loading_layer_ids = None

    @property
    def start_layer(self):
        return self.model.start_layer

    @property
    def end_layer(self):
        return self.model.end_layer

    def finalize_elastic_pp_weight_profile(self):
        if self._elastic_pp_weight_profiler is None:
            raise RuntimeError("elastic PP weight profiling is not enabled")
        return self._elastic_pp_weight_profiler.finalize(self)

    def get_elastic_pp_weight_profile(self):
        if self._elastic_pp_weight_profiler is None:
            raise RuntimeError("elastic PP weight profiling is not enabled")
        return self._elastic_pp_weight_profiler.get_profile()

    @contextmanager
    def elastic_pp_weight_load(self, generation: int, layer_ids: frozenset[int]):
        with self._elastic_pp_weight_load_lock:
            if self._elastic_pp_loading_layer_ids is not None:
                raise RuntimeError("another elastic PP weight load is active")
            self._elastic_pp_loading_layer_ids = layer_ids
            try:
                yield
                self.model.mark_elastic_pp_layers_loaded(generation, set(layer_ids))
            finally:
                self._elastic_pp_loading_layer_ids = None

    def prepare_elastic_pp_layers(self, *, target, generation, device, dtype):
        return self.model.prepare_elastic_pp_layers(
            target=target, generation=generation, device=device, dtype=dtype
        )

    def retain_elastic_pp_graph_layers(self):
        self.model.retain_elastic_pp_graph_layers()

    def export_elastic_pp_layer_state(self, layer_ids):
        return self.model.export_elastic_pp_layer_state(layer_ids)

    def import_elastic_pp_layer_state(self, *, generation, layer_states):
        return self.model.import_elastic_pp_layer_state(
            generation=generation, layer_states=layer_states
        )

    def mark_elastic_pp_layers_loaded(self, *, generation, layer_ids):
        self.model.mark_elastic_pp_layers_loaded(generation, layer_ids)

    def elastic_pp_layers_ready(self, generation):
        return self.model.elastic_pp_layers_ready(generation)

    def commit_elastic_pp_layers(self, *, generation):
        target = self.model.commit_elastic_pp_layers(generation)
        refresh = getattr(self, "_refresh_elastic_pp_moe_weights", None)
        if refresh is not None:
            refresh()
        return target

    def cleanup_elastic_pp_layers(self, *, generation):
        return self.model.take_elastic_pp_cleanup_layers(generation)

    def abort_elastic_pp_layers(self, *, generation):
        return self.model.abort_elastic_pp_layers(generation)
