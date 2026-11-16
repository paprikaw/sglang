"""Shared transactional layer residency for elastic pipeline models."""

from __future__ import annotations

import gc
from collections.abc import Iterable

import torch
from torch import nn

from sglang.srt.elastic_pp.topology import LayerRange
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.model_loader.utils import set_default_torch_dtype


class ElasticPPModelMixin:
    """Prepare, transfer and publish decoder layers without reloading a checkpoint.

    Models supply their layer factory, lock, residency tracker and mutable
    start/end layer boundaries. The forward path must use the same lock.
    """

    elastic_pp_quantization_methods = (None, "compressed_tensors")

    @staticmethod
    def _release_elastic_layer_loader_refs(layers: Iterable[nn.Module]) -> None:
        # Loaders are bound methods or closures capturing their owning module.
        # After warmup SGLang freezes GC, so detaching a retired layer alone
        # leaves module -> parameter -> loader -> module cycles (and all their
        # CUDA storage) alive forever. Only retired/aborted layers reach here;
        # removing loader metadata breaks those cycles without a global GC or
        # mutating shared submodules such as cached rotary embeddings.
        for layer in layers:
            for parameter in layer.parameters():
                parameter.__dict__.pop("weight_loader", None)
                parameter.__dict__.pop("_weight_loader", None)

    def _resident_elastic_layer_ids(self) -> set[int]:
        return {
            layer_id
            for layer_id, layer in enumerate(self.layers)
            if not isinstance(layer, PPMissingLayer)
        }

    def retain_elastic_pp_graph_layers(self) -> None:
        """Pin immutable parameter storage for the explicitly prepared Graph set."""
        if self._elastic_layer_residency is None:
            raise RuntimeError("Graph residency requires elastic model layers")
        self._elastic_graph_layers = {
            i: self.layers[i] for i in self._resident_elastic_layer_ids()
        }

    def enable_elastic_pp_evictable_weights(self):
        """Move weights to stable VA before the first candidate Graph capture."""
        from sglang.srt.elastic_pp.weight_backing import EvictableWeightBank

        if self.quant_config is not None:
            quant_name = self.quant_config.get_name()
            experts = [
                m
                for i in self._resident_elastic_layer_ids()
                for m in self.layers[i].modules()
                if hasattr(m, "num_local_experts")
            ]
            if quant_name != "fp8" or any(
                not getattr(m, "_qwen_a100_fp8_marlin_prepared", False) for m in experts
            ):
                raise ValueError(
                    "nonresident quantized weights require processed Qwen FP8 Marlin layers"
                )
        bank = self._elastic_pp_weight_backing = EvictableWeightBank()
        # VMM allocations bypass PyTorch's allocator. Return loader scratch
        # before the first copy and each converted layer's old blocks before
        # allocating the next, otherwise conversion can reserve a second model.
        torch.cuda.empty_cache()
        for layer_id in sorted(self._resident_elastic_layer_ids()):
            bank.prepare(layer_id, self.layers[layer_id])
            torch.cuda.empty_cache()
        return bank

    @staticmethod
    def _prepare_elastic_quantized_layer(layer: nn.Module) -> None:
        """Put a new quantized layer in its post-load runtime layout.

        Runtime transfer writes the already-processed source state directly
        into the destination parameters.  Seed temporary parameters with safe
        values before invoking the same post-load hooks as the model loader;
        the seeds are never published and are overwritten by the transfer.
        """

        for module in layer.modules():
            quant_method = getattr(module, "quant_method", None)
            if quant_method is None:
                continue
            with torch.no_grad():
                for name, parameter in module.named_parameters(recurse=False):
                    if parameter is None:
                        continue
                    if "scale" in name:
                        parameter.fill_(1)
                    else:
                        parameter.zero_()
                for name, buffer in module.named_buffers(recurse=False):
                    if buffer is None:
                        continue
                    if "scale" in name:
                        buffer.fill_(1)
                    else:
                        buffer.zero_()
            quant_method.process_weights_after_loading(module)

    def prepare_elastic_pp_layers(
        self,
        *,
        target: LayerRange,
        generation: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> frozenset[int]:
        """Materialize missing target layers without changing the live range."""

        try:
            return self._prepare_elastic_pp_layers(
                target=target, generation=generation, device=device, dtype=dtype
            )
        except BaseException as error:
            from sglang.srt.elastic_pp.backing_spill import discard_tensor_tracebacks

            # A retained OOM exception must not pin a partially constructed
            # target layer while the coordinator restores source KV backing.
            # Constructor frames can also contain loader-reference cycles;
            # collect these new, unfrozen objects after dropping the frames.
            discard_tensor_tracebacks(error)
            gc.collect()
            raise error from None

    def _prepare_elastic_pp_layers(
        self,
        *,
        target: LayerRange,
        generation: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> frozenset[int]:
        if self._elastic_layer_residency is None:
            raise RuntimeError("elastic PP layer residency is not enabled")
        quant_name = None if self.quant_config is None else self.quant_config.get_name()
        if quant_name not in self.elastic_pp_quantization_methods:
            raise NotImplementedError(
                "elastic PP runtime layer transfer currently supports "
                "the configured model quantizations, got "
                f"{quant_name!r}"
            )
        with self._elastic_layer_lock:
            preparation = self._elastic_layer_residency.begin_prepare(
                generation=generation,
                target=target,
                resident_layer_ids=self._resident_elastic_layer_ids(),
            )
        if not preparation.new_layer_ids:
            return preparation.new_layer_ids

        create_ids = tuple(
            layer_id
            for layer_id in sorted(preparation.new_layer_ids)
            if isinstance(self.layers[layer_id], PPMissingLayer)
        )
        created: dict[int, nn.Module] = {}
        try:
            with set_default_torch_dtype(dtype), device:
                for layer_id in create_ids:
                    cached = getattr(self, "_elastic_graph_layers", {}).get(layer_id)
                    layer = cached
                    if layer is None:
                        layer = self._elastic_layer_factory(
                            layer_id, target.start
                        ).eval()
                    if cached is None and quant_name is not None:
                        self._prepare_elastic_quantized_layer(layer)
                    created[layer_id] = layer
                    backing = getattr(self, "_elastic_pp_weight_backing", None)
                    if backing is not None:
                        backing.prepare(layer_id, layer)
            if getattr(self, "_elastic_pp_weight_backing", None) is not None:
                torch.cuda.empty_cache()
            with self._elastic_layer_lock:
                for layer_id, layer in created.items():
                    if not isinstance(self.layers[layer_id], PPMissingLayer):
                        raise RuntimeError(
                            "elastic layer slot changed during preparation: "
                            f"layer={layer_id}"
                        )
                    self.layers[layer_id] = layer
        except BaseException:
            backing = getattr(self, "_elastic_pp_weight_backing", None)
            if backing is not None:
                backing.evict(created.keys())
            # Captured layers intentionally remain alive in the Graph bank.
            # Keep their checkpoint loaders so a later CPU-cache restore can
            # retry without replacing addresses after a mapping failure.
            cache = getattr(self, "_elastic_graph_layers", {})
            self._release_elastic_layer_loader_refs(
                layer for layer_id, layer in created.items() if layer_id not in cache
            )
            with self._elastic_layer_lock:
                remove_ids = self._elastic_layer_residency.abort(generation)
                for layer_id in remove_ids:
                    self.layers[layer_id] = PPMissingLayer()
            raise
        return preparation.new_layer_ids

    def mark_elastic_pp_layers_loaded(
        self, generation: int, layer_ids: set[int]
    ) -> None:
        if self._elastic_layer_residency is None:
            raise RuntimeError("elastic PP layer residency is not enabled")
        with self._elastic_layer_lock:
            self._elastic_layer_residency.mark_loaded(generation, layer_ids)

    def elastic_pp_layers_ready(self, generation: int) -> bool:
        if self._elastic_layer_residency is None:
            raise RuntimeError("elastic PP layer residency is not enabled")
        with self._elastic_layer_lock:
            return self._elastic_layer_residency.ready_to_commit(generation)

    def export_elastic_pp_layer_state(
        self, layer_ids: Iterable[int]
    ) -> dict[int, dict[str, torch.Tensor]]:
        """Expose read-only runtime tensors for direct owner-to-owner transfer."""

        with self._elastic_layer_lock:
            exported: dict[int, dict[str, torch.Tensor]] = {}
            for layer_id in layer_ids:
                layer_id = int(layer_id)
                layer = self.layers[layer_id]
                if isinstance(layer, PPMissingLayer):
                    raise ValueError(f"cannot export nonresident layer {layer_id}")
                exported[layer_id] = {
                    name: tensor.detach() for name, tensor in layer.state_dict().items()
                }
            return exported

    def import_elastic_pp_layer_state(
        self,
        *,
        generation: int,
        layer_states: dict[int, dict[str, torch.Tensor]],
    ) -> frozenset[int]:
        """Load transferred runtime tensors into still-inactive target layers."""

        if self._elastic_layer_residency is None:
            raise RuntimeError("elastic PP layer residency is not enabled")
        with self._elastic_layer_lock:
            pending = self._elastic_layer_residency.pending
            if pending is None or pending.generation != generation:
                raise RuntimeError(
                    f"no layer preparation exists for generation {generation}"
                )
            layer_ids = frozenset(int(layer_id) for layer_id in layer_states)
            unexpected = layer_ids - pending.new_layer_ids
            if unexpected:
                raise ValueError(
                    f"received state for unrequested layers: {sorted(unexpected)}"
                )
            for layer_id in sorted(layer_ids):
                layer = self.layers[layer_id]
                if isinstance(layer, PPMissingLayer):
                    raise RuntimeError(f"layer {layer_id} is not materialized")
                layer.load_state_dict(layer_states[layer_id], strict=True, assign=False)
            self._elastic_layer_residency.mark_loaded(generation, set(layer_ids))
            return layer_ids

    def commit_elastic_pp_layers(self, generation: int) -> LayerRange:
        """Atomically publish a prepared layer range at the scheduler fence."""

        if self._elastic_layer_residency is None:
            raise RuntimeError("elastic PP layer residency is not enabled")
        with self._elastic_layer_lock:
            target = self._elastic_layer_residency.commit(generation)
            for layer_id in range(target.start, target.end):
                layer = self.layers[layer_id]
                if isinstance(layer, PPMissingLayer):
                    raise RuntimeError(
                        f"committed elastic layer {layer_id} is not resident"
                    )
                self_attn = getattr(layer, "self_attn", None)
                if self_attn is not None:
                    self_attn.start_layer = target.start
            self.start_layer = target.start
            self.end_layer = target.end
            return target

    def abort_elastic_pp_layers(self, generation: int) -> tuple[nn.Module, ...]:
        if self._elastic_layer_residency is None:
            raise RuntimeError("elastic PP layer residency is not enabled")
        with self._elastic_layer_lock:
            if self._elastic_layer_residency.pending is None:
                if generation <= self._elastic_layer_residency.generation:
                    raise RuntimeError(
                        "cannot abort a published elastic layer generation"
                    )
                # The allocation helper may already have rolled back an OOM.
                return ()
            new_layer_ids = self._elastic_layer_residency.abort(generation)
            backing = getattr(self, "_elastic_pp_weight_backing", None)
            if backing is not None:
                backing.evict(new_layer_ids)
            removed = []
            for layer_id in sorted(new_layer_ids):
                layer = self.layers[layer_id]
                if not isinstance(layer, PPMissingLayer):
                    if layer_id not in getattr(self, "_elastic_graph_layers", {}):
                        removed.append(layer)
                    self.layers[layer_id] = PPMissingLayer()
            self._release_elastic_layer_loader_refs(removed)
            return tuple(removed)

    def take_elastic_pp_cleanup_layers(self, generation: int) -> tuple[nn.Module, ...]:
        """Detach old-generation layers after the target stream has drained."""

        if self._elastic_layer_residency is None:
            raise RuntimeError("elastic PP layer residency is not enabled")
        with self._elastic_layer_lock:
            cleanup_ids = self._elastic_layer_residency.cleanup_candidates(
                generation, self._resident_elastic_layer_ids()
            )
            backing = getattr(self, "_elastic_pp_weight_backing", None)
            if backing is not None:
                backing.evict(cleanup_ids)
            removed = []
            removed_ids = set()
            for layer_id in sorted(cleanup_ids):
                layer = self.layers[layer_id]
                if not isinstance(layer, PPMissingLayer):
                    cache = getattr(self, "_elastic_graph_layers", None)
                    if cache is None:
                        removed.append(layer)
                    else:
                        # The layer is logically detached, but its captured
                        # storage must remain alive when that layout returns.
                        cache[layer_id] = layer
                    removed_ids.add(layer_id)
                    self.layers[layer_id] = PPMissingLayer()
            self._elastic_layer_residency.finish_cleanup(generation, removed_ids)
            self._release_elastic_layer_loader_refs(removed)
            return tuple(removed)

    def get_elastic_pp_layer_range(self) -> LayerRange:
        if self._elastic_layer_residency is None:
            return LayerRange(self.start_layer, self.end_layer)
        return self._elastic_layer_residency.scheduled_range
