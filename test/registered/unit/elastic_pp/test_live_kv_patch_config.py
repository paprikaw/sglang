from types import SimpleNamespace

import pytest

from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def args(**overrides):
    options = dict(
        model_path="unused",
        pp_size=3,
        enable_elastic_pp=True,
        elastic_pp_initial_partition="20,0,20",
        disable_cuda_graph=True,
        elastic_pp_transition_strategy="background",
        enable_elastic_pp_kv_patch=True,
    )
    options.update(overrides)
    result = ServerArgs(**options)
    result.get_model_config = lambda: SimpleNamespace(
        hf_config=SimpleNamespace(architectures=["LlamaForCausalLM"])
    )
    return result


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"enable_elastic_pp": False}, "requires --enable-elastic-pp"),
        ({"elastic_pp_transition_strategy": "sync"}, "background/overlap"),
        ({"elastic_pp_transition_backing": "cpu_spill"}, "background/overlap"),
        ({"elastic_pp_kv_backend": "kvcached"}, "capacity frontier"),
        ({"speculative_algorithm": "EAGLE"}, "no speculative"),
    ],
)
def test_unqualified_patch_combinations_rejected(overrides, message):
    with pytest.raises(ValueError, match=message):
        args(**overrides)._handle_pipeline_parallelism()


@pytest.mark.parametrize("page_size", [1, 64])
def test_native_eager_background_patch_and_default_sync_remain_accepted(page_size):
    args(page_size=page_size)._handle_pipeline_parallelism()
    args(
        enable_elastic_pp_kv_patch=False, elastic_pp_transition_strategy="sync"
    )._handle_pipeline_parallelism()


def test_normalized_partition_reaches_projected_configuration():
    config = args(elastic_pp_initial_partition="20, 0, 20")
    config._handle_pipeline_parallelism()
    assert config._resolved().elastic_pp_initial_partition == "20,0,20"


@pytest.mark.parametrize(
    "overrides, message",
    [
        (
            {"enable_elastic_pp_kv_patch": False},
            "requires --enable-elastic-pp-kv-patch",
        ),
        ({"enable_hierarchical_cache": False}, "requires HiCache"),
    ],
)
def test_nixl_keeps_explicit_correctness_scope(overrides, message):
    options = dict(elastic_pp_kv_patch_transport="nixl", enable_hierarchical_cache=True)
    options.update(overrides)
    with pytest.raises(ValueError, match=message):
        args(**options)._handle_pipeline_parallelism()


def test_nixl_selection_survives_argument_projection():
    config = args(elastic_pp_kv_patch_transport="nixl", enable_hierarchical_cache=True)
    config._handle_pipeline_parallelism()
    assert config.elastic_pp_kv_patch_transport == "nixl"


@pytest.mark.parametrize("transport", ["cpu", "nixl"])
def test_native_graph_online_patch_is_accepted(transport):
    args(
        disable_cuda_graph=False,
        enable_hierarchical_cache=True,
        elastic_pp_kv_patch_transport=transport,
    )._handle_pipeline_parallelism()


def test_elastic_graph_rejects_unqualified_compile():
    with pytest.raises(ValueError, match="no torch.compile"):
        args(
            disable_cuda_graph=False, enable_torch_compile=True
        )._handle_pipeline_parallelism()
