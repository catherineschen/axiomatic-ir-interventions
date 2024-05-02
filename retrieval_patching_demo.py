import torch

from transformers import AutoTokenizer, AutoModel
import transformer_lens.utils as utils
from transformer_lens import HookedEncoder

from patching_helpers import (
    get_act_patch_block_every,
    get_act_patch_attn_head_out_all_pos,
    get_act_patch_attn_head_by_pos
)


def load_tokenizer_and_models(hf_model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    hf_model = AutoModel.from_pretrained(hf_model_name)
    hf_model.to(device)

    tl_model = HookedEncoder.from_pretrained(hf_model_name, device=device, hf_model=hf_model)

    return tokenizer, tl_model


def main():
    torch.set_grad_enabled(False)
    device = utils.get_device()

    # Load and initialize models
    pre_trained_model_name = "sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco"
    tokenizer, tl_model = load_tokenizer_and_models(pre_trained_model_name, device)

    # Minimal usage example - note that the document is not super relvant to show the difference in scores after perturbation
    query = "what is the acceptance rate at wellesley"
    original_doc = "College acceptance rates in the USA are becoming more competitive."
    perturbed_doc = original_doc + " " + "wellesley"
    
    # Tokenize query and documents, adjusting for mismatched lengths if necessary
    tokenized_query = tokenizer(query, return_tensors="pt")
    tokenized_baseline_doc = tokenizer(original_doc, return_tensors="pt")
    tokenized_p_doc = tokenizer(perturbed_doc, return_tensors="pt")

    filler_token = tokenizer.encode("a", add_special_tokens=False)[0]
    b_len = torch.sum(tokenized_baseline_doc["attention_mask"]).item()
    p_len = torch.sum(tokenized_p_doc["attention_mask"]).item()
    if b_len != p_len:
        adj_n = p_len - b_len
        cls_tok = tokenized_baseline_doc["input_ids"][0][0]
        sep_tok = tokenized_baseline_doc["input_ids"][0][-1]
        filler_tokens = torch.full((adj_n,), filler_token)
        filler_attn_mask = torch.full((adj_n,), tokenized_baseline_doc["attention_mask"][0][1]) 
        adj_doc = torch.cat((tokenized_baseline_doc["input_ids"][0][1:-1], filler_tokens))
        tokenized_baseline_doc["input_ids"] = torch.cat((cls_tok.view(1), adj_doc, sep_tok.view(1)), dim=0).view(1,-1)
        tokenized_baseline_doc["attention_mask"] = torch.cat((tokenized_baseline_doc["attention_mask"][0], filler_attn_mask), dim=0).view(1,-1)

    # Get query embedding
    q_outputs = tl_model(
        tokenized_query["input_ids"],
        return_type="embeddings",
        one_zero_attention_mask=tokenized_query["attention_mask"],
    )
    q_embedding = q_outputs[:,0,:].squeeze(0)

    # Run model on baseline document + calculate score
    baseline_outputs = tl_model(
        tokenized_baseline_doc["input_ids"],
        return_type="embeddings",
        one_zero_attention_mask=tokenized_baseline_doc["attention_mask"],
    )
    baseline_embedding = baseline_outputs[:,0,:].squeeze(0)
    baseline_score = torch.matmul(q_embedding, baseline_embedding.t())
    
    # Run model on perturbed document + calculate score + cache activations
    perturbed_outputs, perturbed_cache = tl_model.run_with_cache(
        tokenized_p_doc["input_ids"],
        one_zero_attention_mask=tokenized_p_doc["attention_mask"],
        return_type="embeddings",
    )
    perturbed_embedding = perturbed_outputs[:,0,:].squeeze(0)
    perturbed_score = torch.matmul(q_embedding, perturbed_embedding.t())

    '''
    Linear function of score diff, calibrated so that it equals 0 when performance is 
    same as on clean input, and 1 when performance is same as on corrupted input.
    '''
    def ranking_metric(patched_doc_embedding, og_score=baseline_score, p_score=perturbed_score):
        patched_score = torch.matmul(q_embedding, patched_doc_embedding.t())
        return (patched_score - og_score) / (p_score - og_score)
                

    # Activation patching - by block over all token positions (e.g., input to residual stream, activation block output, MLP output)
    act_patch_block_every = get_act_patch_block_every(
        tl_model,
        device,
        tokenized_baseline_doc,
        perturbed_cache,
        ranking_metric
    )

    # Activation patching - by attention head over all token positions
    act_patch_attn_head_out_all_pos = get_act_patch_attn_head_out_all_pos(
        tl_model,
        device,
        tokenized_baseline_doc,
        perturbed_cache,
        ranking_metric
    )

    # Activation patching - by attention head by individual token position
    layer_head_list = [(0,9), (1,6), (2,3), (3,8)] # heads to patch
    act_patch_attn_head_out_by_pos = get_act_patch_attn_head_by_pos(
        tl_model,
        device,
        tokenized_baseline_doc,
        perturbed_cache,
        ranking_metric,
        layer_head_list
    )

    return

if __name__ == "__main__":
    main()
