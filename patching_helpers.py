import torch
from tqdm import tqdm

from transformer_lens import HookedEncoder, ActivationCache
from transformer_lens import patching
import transformer_lens.utils as utils

from jaxtyping import Float
from typing import Callable
from functools import partial

'''
    Patches a given sequence position in the residual stream, using the value
    from the clean cache.
'''
def patch_residual_component(
    corrupted_component: Float[torch.Tensor, "batch pos d_model"],
    hook,
    pos,
    clean_cache,
):
    corrupted_component[:, pos, :] = clean_cache[hook.name][:, pos, :]
    return corrupted_component


'''
Returns an array of results of patching each position at each layer in the residual
stream, using the value from the clean cache.

The results are calculated using the patching_metric function, which should be
called on the model's logit output.
'''
def get_act_patch_block_every(
    model: HookedEncoder, 
    device,
    corrupted_tokens: Float[torch.Tensor, "batch pos"], 
    clean_cache: ActivationCache, 
    patching_metric: Callable[[Float[torch.Tensor, "batch pos d_vocab"]], float]
) -> Float[torch.Tensor, "3 layer pos"]:

    model.reset_hooks()
    _, seq_len = corrupted_tokens["input_ids"].size()
    results = torch.zeros(3, model.cfg.n_layers, seq_len, device=device, dtype=torch.float32)

    # send tokens to device if not already there
    corrupted_tokens["input_ids"] = corrupted_tokens["input_ids"].to(device)
    corrupted_tokens["attention_mask"] = corrupted_tokens["attention_mask"].to(device)

    for component_idx, component in enumerate(["resid_pre", "attn_out", "mlp_out"]):
        print("Patching:", component)
        for layer in tqdm(range(model.cfg.n_layers)):
            for position in range(seq_len):
                hook_fn = partial(patch_residual_component, pos=position, clean_cache=clean_cache)
                patched_outputs = model.run_with_hooks(
                    corrupted_tokens["input_ids"],
                    one_zero_attention_mask=corrupted_tokens["attention_mask"],
                    return_type="embeddings",
                    fwd_hooks = [(utils.get_act_name(component, layer), hook_fn)],
                )
                patched_embedding = patched_outputs[:,0,:].squeeze(0)
                results[component_idx, layer, position] = patching_metric(patched_embedding)

    return results

'''
Patches the output of a given head (before it's added to the residual stream) at
every sequence position, using the value from the clean cache.
'''
def patch_head_vector(
    corrupted_head_vector: Float[torch.Tensor, "batch pos head_index d_head"],
    hook, #: HookPoint, 
    head_index: int, 
    clean_cache: ActivationCache
) -> Float[torch.Tensor, "batch pos head_index d_head"]:
 
    corrupted_head_vector[:, :, head_index] = clean_cache[hook.name][:, :, head_index]
    return corrupted_head_vector


'''
Returns an array of results of patching at all positions for each head in each
layer, using the value from the clean cache.

The results are calculated using the patching_metric function, which should be
called on the model's embedding output.
'''
def get_act_patch_attn_head_out_all_pos(
    model: HookedEncoder, 
    device,
    corrupted_tokens: Float[torch.Tensor, "batch pos"], 
    clean_cache: ActivationCache, 
    patching_metric: Callable
) -> Float[torch.Tensor, "layer head"]:

    model.reset_hooks()
    results = torch.zeros(model.cfg.n_layers, model.cfg.n_heads, device=device, dtype=torch.float32)

    print("Patching: attn_heads")
    for layer in tqdm(range(model.cfg.n_layers)):
        for head in range(model.cfg.n_heads):
            hook_fn = partial(patch_head_vector, head_index=head, clean_cache=clean_cache)
            patched_outputs = model.run_with_hooks(
                corrupted_tokens["input_ids"],
                one_zero_attention_mask=corrupted_tokens["attention_mask"],
                return_type="embeddings",
                fwd_hooks = [(utils.get_act_name("z", layer), hook_fn)],
            )
            patched_embedding = patched_outputs[:,0,:].squeeze(0)
            results[layer, head] = patching_metric(patched_embedding)

    return results


def patch_head_vector_by_pos_pattern(
    corrupted_activation: Float[torch.Tensor, "batch pos head_index pos_q pos_k"],
    hook, #: HookPoint, 
    pos,
    head_index: int, 
    clean_cache: ActivationCache
) -> Float[torch.Tensor, "batch pos head_index d_head"]:

    corrupted_activation[:,head_index,pos,:] = clean_cache[hook.name][:,head_index,pos,:]
    return corrupted_activation


def patch_head_vector_by_pos(
    corrupted_activation: Float[torch.Tensor, "batch pos head_index d_head"],
    hook, #: HookPoint, 
    pos,
    head_index: int, 
    clean_cache: ActivationCache
) -> Float[torch.Tensor, "batch pos head_index d_head"]:

    corrupted_activation[:, pos, head_index] = clean_cache[hook.name][:, pos, head_index]
    return corrupted_activation


def get_act_patch_attn_head_by_pos(
    model: HookedEncoder, 
    device,
    corrupted_tokens: Float[torch.Tensor, "batch pos"], 
    clean_cache: ActivationCache, 
    patching_metric: Callable,
    layer_head_list,
) -> Float[torch.Tensor, "layer pos head"]:
    
    model.reset_hooks()
    _, seq_len = corrupted_tokens["input_ids"].size()
    results = torch.zeros(2, len(layer_head_list), seq_len, device=device, dtype=torch.float32)

    for component_idx, component in enumerate(["z", "pattern"]):
        for i, layer_head in enumerate(layer_head_list):
            layer = layer_head[0]
            head = layer_head[1]
            for position in range(seq_len):
                patch_fn = patch_head_vector_by_pos_pattern if component == "pattern" else patch_head_vector_by_pos
                hook_fn = partial(patch_fn, pos=position, head_index=head, clean_cache=clean_cache)
                patched_outputs = model.run_with_hooks(
                    corrupted_tokens["input_ids"],
                    one_zero_attention_mask=corrupted_tokens["attention_mask"],
                    return_type="embeddings",
                    fwd_hooks = [(utils.get_act_name(component, layer), hook_fn)],
                )
                patched_embedding = patched_outputs[:,0,:].squeeze(0)
                results[component_idx, i, position] = patching_metric(patched_embedding)

    return results
