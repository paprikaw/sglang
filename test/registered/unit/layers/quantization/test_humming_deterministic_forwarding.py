"""CPU contract tests for the native deterministic -> Humming bridge.

Execute the production config-building statements in isolation: importing the
full quantization/runner modules would require native CUDA dependencies even
though this regression only concerns CPU configuration. No Humming kernels run.
"""

import ast
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

ROOT = Path(__file__).resolve().parents[5]
DENSE_PATH = ROOT / "python/sglang/srt/layers/quantization/humming.py"
MOE_PATH = ROOT / "python/sglang/srt/layers/moe/moe_runner/humming.py"


def _environment(enabled):
    return {
        "json": json,
        "envs": SimpleNamespace(
            SGLANG_HUMMING_USE_F16_ACCUM=SimpleNamespace(get=lambda: False)
        ),
        "get_exec": lambda: SimpleNamespace(
            deterministic=SimpleNamespace(enable_deterministic_inference=enabled)
        ),
    }


@pytest.mark.parametrize("enabled", [False, True])
def test_dense_forwards_native_mode_to_actual_compute_config(enabled):
    tree = ast.parse(DENSE_PATH.read_text())
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "process_weights_after_loading"
        and any(
            isinstance(child, ast.Name) and child.id == "compute_config"
            for child in ast.walk(node)
        )
    )
    start = next(
        index
        for index, node in enumerate(method.body)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "compute_config"
                for target in node.targets)
    )
    namespace = _environment(enabled) | {"self": SimpleNamespace()}
    # Include serialization/assignment, not a copied dictionary from this test.
    body = ast.Module(body=method.body[start:], type_ignores=[])
    exec(compile(ast.fix_missing_locations(body), str(DENSE_PATH), "exec"), namespace)
    config = json.loads(namespace["self"].compute_config)
    assert config == {
        "gemm_type": "dense",
        "use_f16_accum": False,
        "use_batch_invariant": enabled,
    }


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("gemm_type", ["grouped_masked", "indexed"])
def test_moe_forwards_native_mode_to_compute_and_both_explicit_tunings(
    enabled, gemm_type
):
    tree = ast.parse(MOE_PATH.read_text())
    method = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "get_humming_gemm_configs"
    )
    method.returns = None
    for argument in method.args.args:
        argument.annotation = None
    tuning = Mock(side_effect=lambda **kwargs: [{
        "sublayer": kwargs["sublayer_name"],
        "use_stream_k": not kwargs["use_batch_invariant"],
    }])
    namespace = _environment(enabled) | {
        "HummingMethod": SimpleNamespace(get_default_tuning_configs=tuning)
    }
    exec(compile(ast.fix_missing_locations(ast.Module(
        body=[method], type_ignores=[]
    )), str(MOE_PATH), "exec"), namespace)
    layer = object()
    runner = SimpleNamespace(layer=layer, humming_gemm_configs={})
    kind = SimpleNamespace(value=gemm_type)
    result = namespace[method.name](runner, kind)
    assert json.loads(result["compute_config_str"])["use_batch_invariant"] is enabled
    assert result["compute_config"]["gemm_type"] == gemm_type
    assert tuning.call_count == 2
    for call, sublayer in zip(tuning.call_args_list, ["w13", "w2"]):
        assert call.kwargs == dict(
            layer=layer,
            use_f16_accum=False,
            use_batch_invariant=enabled,
            gemm_type=kind,
            sublayer_name=sublayer,
        )
        assert json.loads(result[f"{sublayer}_tuning_config_str"])[0][
            "use_stream_k"
        ] is not enabled
    assert namespace[method.name](runner, kind) is result
    assert tuning.call_count == 2


@pytest.mark.parametrize("path", [DENSE_PATH, MOE_PATH])
def test_production_modules_import_the_runtime_accessor(path):
    tree = ast.parse(path.read_text())
    assert any(
        isinstance(node, ast.ImportFrom)
        and node.module == "sglang.srt.runtime_context"
        and any(name.name == "get_exec" for name in node.names)
        for node in tree.body
    )
