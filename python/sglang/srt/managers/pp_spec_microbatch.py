"""Bounded, opt-in two-slot PP2/MTP scheduling primitives.

Only independent requests overlap: a slot still waits for its own accepted
tokens and next draft before reuse. The transport uses distinct activation and
feedback process groups so a pending activation cannot block its own feedback.
"""

from typing import Any, Callable, Dict, Optional

import torch


def validate_microbatch_config(
    *, enabled, pp_size, pp_async_batch_depth, pp_spec_enabled,
    speculative_algorithm, topk, overlap_disabled, elastic,
    disaggregation_mode, dp_attention,
):
    if not enabled:
        return
    if not (
        pp_size == 2 and pp_async_batch_depth == 0 and pp_spec_enabled
        and speculative_algorithm == "EAGLE" and topk == 1
        and overlap_disabled and not elastic
        and disaggregation_mode == "null" and not dp_attention
    ):
        raise ValueError(
            "SGLANG_PP_SPEC_MICROBATCH requires PP2, async depth 0, "
            "SGLANG_ENABLE_PP_SPEC=1, single-layer EAGLE topk1, "
            "--disable-overlap-schedule, no DP Attention, elastic PP or PD"
        )


def own_payload(tensor_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Snapshot graph-owned outputs on their producer stream before recording
    the send-ready event. P2PWork then holds these copies until completion.
    """
    return {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in tensor_dict.items()
    }


def launch_then_exchange(
    launch: Optional[Callable], send_proxy: Callable, exchange: Callable,
):
    """Submit the ready slot before retiring the *other* slot's feedback.

    send_proxy must enqueue on the activation group and retain its payload.
    exchange uses the independent feedback group and existing slot/RID protocol.
    Empty slots still exchange feedback so startup and draining make progress.
    """
    result = None
    if launch is not None:
        result = launch()
        send_proxy(result)
    return result, exchange()
