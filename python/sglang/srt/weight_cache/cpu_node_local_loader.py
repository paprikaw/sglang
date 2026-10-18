"""Normal Hy3 loading from an explicitly selected, prebuilt CPU RAM bank."""

import copy
import hashlib
import json
import logging
import os
import time
from pathlib import Path

try:
    from .cpu_node_local import describe_model, model_view
except ImportError:
    from cpu_node_local import describe_model, model_view

logger = logging.getLogger(__name__)


class NodeLocalCpuModelLoader:
    def __init__(self, load_config, fallback):
        self.load_config = load_config
        self.fallback = fallback

    def _arguments(self, model_config):
        path = os.environ["SGLANG_NODE_LOCAL_WEIGHT_CACHE_SELECTION_MANIFEST"]
        expected = os.environ.get("SGLANG_NODE_LOCAL_WEIGHT_CACHE_SELECTION_SHA256", "")
        if (
            len(expected) != 64
            or hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected
        ):
            raise ValueError(
                "node-local selection manifest hash differs from launch binding"
            )
        return (
            model_config.model_path,
            os.environ["SGLANG_NODE_LOCAL_WEIGHT_CACHE_SOCKET"],
            os.environ.get("SGLANG_NODE_LOCAL_WEIGHT_CACHE_SOURCE_COMMIT", ""),
            path,
        )

    @staticmethod
    def _validate_model_config(model_config, selection):
        config = model_config.hf_config
        if (
            getattr(config, "model_type", None) != "hy_v3"
            or getattr(config, "num_hidden_layers", None)
            != selection["effective_layers"]
            or getattr(config, "num_nextn_predict_layers", None) != 0
            or getattr(model_config, "quantization", None) is not None
            or str(model_config.dtype).replace("torch.", "") != "bfloat16"
        ):
            raise ValueError(
                "model dtype/layers/MTP/quantization differs from node-local Hy3 selection"
            )
        source = json.loads(
            (Path(selection["model_root"]) / "config.json").read_bytes()
        )
        for key in (
            "architectures",
            "hidden_size",
            "intermediate_size",
            "moe_intermediate_size",
            "expert_hidden_dim",
            "first_k_dense_replace",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "num_experts",
            "num_experts_per_tok",
            "num_shared_experts",
            "tie_word_embeddings",
            "qk_norm",
            "hidden_act",
        ):
            if key in source and getattr(config, key, None) != source[key]:
                raise ValueError(
                    f"model architecture override {key} differs from node-local source"
                )

    @staticmethod
    def _validate_topology(selection):
        from sglang.srt.runtime_context import get_exec, get_parallel

        parallel = get_parallel()
        moe = get_exec().moe
        if (
            moe.enable_eplb
            or moe.ep_join_mode is not None
            or moe.ep_num_redundant_experts != 0
            or moe.init_expert_location != "trivial"
        ):
            raise ValueError(
                "node-local bank forbids elastic/redundant/rearranged expert placement"
            )
        expected = {"attn_tp_size": 4, "moe_tp_size": 1, "moe_dp_size": 1}
        if selection["layout"] == "ep8":
            expected.update(tp_size=8, pp_size=1, moe_ep_size=8, attn_dp_size=2)
            owned = (
                selection["node_rank"] * 4
                <= parallel.moe_ep_rank
                < (selection["node_rank"] + 1) * 4
            )
        else:
            expected.update(tp_size=4, pp_size=2, moe_ep_size=4, attn_dp_size=1)
            owned = parallel.pp_rank == selection["node_rank"]
        if not owned or any(
            getattr(parallel, key) != value for key, value in expected.items()
        ):
            raise ValueError(
                "live rank topology differs from node-local weight selection"
            )

    def download_model(self, model_config):
        root, socket, commit, path = self._arguments(model_config)
        bank = describe_model(root, path)
        self._validate_model_config(model_config, bank["selection"])
        with model_view(root, socket, commit, selection_manifest_path=path):
            pass

    def load_model(self, *, model_config, device_config):
        from sglang.srt.runtime_context import get_model

        flags = get_model()
        if flags.weight_loader_disable_mmap:
            raise ValueError("node-local CPU bank requires mmap checkpoint loading")
        root, socket, commit, path = self._arguments(model_config)
        original_extra = self.load_config.model_loader_extra_config
        # Keep iteration synchronous, so no worker/prefetch thread can outlive
        # the authenticated reader pin or temporary selected index.
        scoped_extra = dict(original_extra or {}, enable_multithread_load=False)
        started = time.monotonic()
        with model_view(root, socket, commit, selection_manifest_path=path) as (
            view,
            bank,
        ):
            self._validate_model_config(model_config, bank["selection"])
            self._validate_topology(bank["selection"])
            view_config = copy.copy(model_config)
            view_config.model_path = view
            self.load_config._node_local_weight_cache_view = view
            self.load_config.model_loader_extra_config = scoped_extra
            try:
                with flags.override(weight_loader_prefetch_checkpoints=False):
                    model = self.fallback.load_model(
                        model_config=view_config, device_config=device_config
                    )
            finally:
                view_config.model_path = model_config.model_path
                self.load_config.model_loader_extra_config = original_extra
                del self.load_config._node_local_weight_cache_view
            model._node_local_cpu_weight_cache_key = bank["group_key"]
            model._node_local_cpu_weight_cache_selection = {
                "selection_sha256": bank["selection"]["selection_sha256"],
                "selected_tensors_sha256": bank["selection"]["selected_tensors_sha256"],
                "selected_tensor_count": bank["selection"]["selected_tensor_count"],
            }
        logger.info(
            "CPU weight cache load: source=node_local_cpu_daemon key=%s elapsed=%.3fs",
            bank["group_key"],
            time.monotonic() - started,
        )
        return model
