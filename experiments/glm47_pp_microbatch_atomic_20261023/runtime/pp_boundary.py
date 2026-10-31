"""Align full PP activation buffers with GLM's stage-local scatter boundaries."""
def apply_boundaries(model,full_mode):
    changed=[]
    if not model.pp_group.is_first_rank:
        layer=model.layers[model.start_layer]
        layer.layer_scatter_modes.layer_input_mode=full_mode
        layer.layer_communicator._post_init_communicate()
        changed.append(dict(layer=model.start_layer,direction='input',mode=str(full_mode)))
    if not model.pp_group.is_last_rank:
        layer=model.layers[model.end_layer-1]
        layer.layer_scatter_modes.layer_output_mode=full_mode
        layer.layer_communicator.is_last_layer=True
        layer.layer_communicator._post_init_communicate()
        layer.layer_communicator._communicate_summable_tensor_pair_fn=gather_pp_pair
        changed.append(dict(layer=model.end_layer-1,direction='output',mode=str(full_mode)))
    return changed


def gather_pp_pair(hidden_states,residual,forward_batch,context,**kwargs):
    import torch
    from sglang.srt.layers.dp_attention import attn_tp_all_gather_into_tensor
    assert residual is not None and residual.shape==hidden_states.shape
    shape=(hidden_states.shape[0]*context.attn_tp_size,hidden_states.shape[1])
    full_hidden=torch.empty(shape,dtype=hidden_states.dtype,device=hidden_states.device)
    full_residual=torch.empty_like(full_hidden)
    attn_tp_all_gather_into_tensor(full_hidden,hidden_states)
    attn_tp_all_gather_into_tensor(full_residual,residual)
    return full_hidden,full_residual
