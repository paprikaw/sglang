"""Qwen3.5 hybrid Decode operations for the native two-batch scheduler.

Attention/GDN and Expert kernels are unchanged. Each child owns its backend
metadata and dispatcher handle; only independent request rows are interleaved.
"""
from sglang.srt.batch_overlap.operations import YieldOperation
from sglang.srt.batch_overlap.operations_strategy import OperationsStrategy
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_forward, get_parallel


class HybridLayerOperations:
    def __init__(self, layer):
        self.layer = layer
        self.mlp = layer.mlp

    def prepare_attn(self, state, positions, hidden_states, forward_batch, residual,
                     tbo_subbatch_index=None):
        state.hidden_states_after_comm_pre_attn, state.residual_after_input_ln = (
            self.layer.layer_communicator.prepare_attn(hidden_states, residual, forward_batch)
        )
        state.update(dict(positions=positions, forward_batch=forward_batch,
                          tbo_subbatch_index=tbo_subbatch_index))

    def attn(self, state):
        hidden_states = state.pop("hidden_states_after_comm_pre_attn")
        if hidden_states.shape[0] and not state.forward_batch.forward_mode.is_idle():
            if hasattr(self.layer, "linear_attn"):
                hidden_states = self.layer.linear_attn(hidden_states, state.forward_batch)
            else:
                hidden_states = self.layer.self_attention(
                    positions=state.positions, hidden_states=hidden_states,
                    forward_batch=state.forward_batch)
        state.hidden_states_after_attn = hidden_states

    def prepare_mlp(self, state):
        state.hidden_states_mlp_input, state.residual_after_comm_pre_mlp = (
            self.layer.layer_communicator.prepare_mlp(
                state.pop("hidden_states_after_attn"), state.pop("residual_after_input_ln"),
                state.forward_batch)
        )

    def gate(self, state):
        if state.hidden_states_mlp_input.shape[0]:
            state.router_logits, _ = self.mlp.gate(state.hidden_states_mlp_input)
        else:
            state.router_logits = None

    def select(self, state):
        logits = state.pop("router_logits")
        hidden = state.hidden_states_mlp_input
        if logits is None:
            state.topk_output = self.mlp.topk.empty_topk_output(hidden.device)
        else:
            with get_global_expert_distribution_recorder().with_current_layer(self.layer.layer_id):
                state.topk_output = self.mlp.topk(
                    hidden, logits, num_token_non_padded=state.forward_batch.num_token_non_padded,
                    expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(
                        layer_id=self.layer.layer_id))

    def dispatch_a(self, state):
        self.mlp.experts.dispatcher.dispatch_a(
            hidden_states=state.hidden_states_mlp_input, topk_output=state.pop("topk_output"),
            tbo_subbatch_index=state.tbo_subbatch_index)

    def shared(self, state):
        hidden = state.pop("hidden_states_mlp_input")
        state.shared_output = self.mlp._forward_shared_experts(hidden) if hidden.shape[0] else None

    def dispatch_b(self, state):
        with get_global_expert_distribution_recorder().with_current_layer(self.layer.layer_id):
            state.dispatch_output = self.mlp.experts.dispatcher.dispatch_b(
                tbo_subbatch_index=state.tbo_subbatch_index)

    def experts(self, state):
        communicator = self.layer.layer_communicator
        # This first compatibility path retains the validated Attention TP1
        # communicator behavior; it does not introduce norm/AR fusion changes.
        assert not communicator.should_fuse_mlp_allreduce_with_next_layer(state.forward_batch)
        with get_forward().scoped(
            fuse_mlp_allreduce=False,
            mlp_reduce_scatter=communicator.should_use_reduce_scatter(state.forward_batch)):
            state.combine_input = self.mlp.experts.run_moe_core(dispatch_output=state.dispatch_output)

    def combine_a(self, state):
        self.mlp.experts.dispatcher.combine_a(
            combine_input=state.pop("combine_input"), tbo_subbatch_index=state.tbo_subbatch_index)
        state.pop("dispatch_output")

    def combine_b(self, state):
        state.hidden_states_after_combine = self.mlp.experts.dispatcher.combine_b(
            tbo_subbatch_index=state.tbo_subbatch_index)

    def output(self, state):
        hidden = state.pop("hidden_states_after_combine")
        shared = state.pop("shared_output")
        if shared is not None:
            hidden.add_(shared)
        state.hidden_states_mlp_output = hidden

    def postprocess(self, state):
        hidden, residual = self.layer.layer_communicator.postprocess_layer(
            state.pop("hidden_states_mlp_output"), state.pop("residual_after_comm_pre_mlp"),
            state.forward_batch)
        output = dict(positions=state.positions, hidden_states=hidden, residual=residual,
                      forward_batch=state.forward_batch, tbo_subbatch_index=state.tbo_subbatch_index)
        state.clear(expect_keys={"positions", "forward_batch", "tbo_subbatch_index"})
        return output


def build_strategy(layers, forward_mode):
    if forward_mode != ForwardMode.DECODE:
        raise NotImplementedError("Qwen3.5 hybrid TBO currently supports Decode only")
    assert get_parallel().attn_tp_size == 1
    assert get_parallel().moe_ep_size > 1
    operations = []
    for layer in layers:
        op = HybridLayerOperations(layer)
        operations.extend([
            op.prepare_attn, op.attn, op.prepare_mlp, op.gate, op.select,
            YieldOperation(),
            op.dispatch_a, op.shared,
            YieldOperation(),
            op.dispatch_b, op.experts, op.combine_a,
            YieldOperation(),
            op.combine_b, op.output, op.postprocess,
            YieldOperation(),
        ])
    return OperationsStrategy(operations=operations, deep_gemm_num_sms=None, tbo_delta_stages=1)
