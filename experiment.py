import torch
import numpy as np
import pandas as pd

import os
from tqdm import tqdm
import random
import glob
import argparse

from functools import partial
import transformer_lens.utils as utils
from transformer_lens import patching
from jaxtyping import Float

from helpers import (
    load_json_file,
    load_tokenizer_and_models,
    preprocess_queries,
    preprocess_corpus,
    encode_hf,
    encode_tl,
    compute_ranking_scores
)

from patching_helpers import (
    get_act_patch_block_every,
    get_act_patch_attn_head_out_all_pos,
    get_act_patch_attn_head_by_pos
)


def run_experiment(experiment_type, perturb_type):
    """
        - experiment_type: what type of experiment to run (options below)
            - block : patches the residual stream (before each layer), attention block, and MLP outputs across individual tokens
            - head_all : patches all attention heads individually across all tokens
            - head_pos : patches all attention heads individualls across individual tokens
            - labels : generates the tokenized documents
        - perturb_type : 
            - prepend : additional term is injected at the beginning of the document
            - append : additional term is injected at the end of the document
    """
    torch.set_grad_enabled(False)
    device = utils.get_device()

    use_reduced_dataset = False
    n_queries = 1
    n_docs = 100

    # Load queries and docs from files
    fbase_path = "data"

    # Load files
    tfc1_precomputed_scores = pd.read_csv(os.path.join(fbase_path, "tfc1_add_{}_target_qids_scores.csv".format(perturb_type)))
    tfc1_add_queries = pd.read_csv(os.path.join(fbase_path, "tfc1_add_qids_with_text.csv"), header=None, names=["_id", "text"])
    tfc1_add_baseline_corpus = load_json_file(os.path.join(fbase_path, "tfc1_add_baseline_final_dd_corpus.json"))["corpus"]
    tfc1_add_dd_corpus = load_json_file(os.path.join(fbase_path, "tfc1_add_{}_final_dd_corpus.json".format(perturb_type)))["corpus"]

    target_qids = tfc1_add_queries["_id"].tolist()
    tfc1_add_queries = tfc1_add_queries[tfc1_add_queries["_id"].isin(target_qids)]
    if use_reduced_dataset:
        target_qids = random.sample(tfc1_add_queries["_id"].tolist(), n_queries)
        tfc1_add_queries = tfc1_add_queries[tfc1_add_queries["_id"].isin(target_qids)]

    pre_trained_model_name = "sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco"
    tokenizer, _, tl_model = load_tokenizer_and_models(pre_trained_model_name, device)

    # Preprocess queries
    queries_dataloader = preprocess_queries(tfc1_add_queries, tokenizer)

    # Loop through each query and run activation patching
    for i, qid in enumerate(tqdm(target_qids)):
        print("QID at:", i)

        # Get query embedding
        q_tokenized = list(filter(lambda item: item["_id"] == qid, queries_dataloader))[0]
        q_outputs = tl_model(
            q_tokenized["input_ids"],
            return_type="embeddings",
            one_zero_attention_mask=q_tokenized["attention_mask"],
        )
        q_embedding = q_outputs[:,0,:].squeeze(0) #.detach().cpu().numpy() # leave on device
        
        # Get and preprocess documents
        target_docs = tfc1_add_dd_corpus[str(qid)]
        if use_reduced_dataset:
            target_doc_ids = random.sample(target_docs.keys(), n_docs)
            target_docs = {doc_id: target_docs[doc_id] for doc_id in target_doc_ids}
        corpus_dataloader = preprocess_corpus(target_docs, tokenizer)

        # batch size = 1 doc
        for j, batch in enumerate(corpus_dataloader):
            print("DOC AT:", j)
            try:
                # Get baseline doc
                doc_id = batch["_id"][0]
                baseline_doc = tfc1_add_baseline_corpus[str(qid)][doc_id]["text"]
                baseline_tokens = tokenizer(baseline_doc, truncation=True, return_tensors="pt")

                # Run perturbed prompt with cache to store activations
                perturbed_embeddings, perturbed_cache = tl_model.run_with_cache(
                    batch["input_ids"],
                    one_zero_attention_mask=batch["attention_mask"],
                    return_type="embeddings",
                )

                # Check lengths of pertubred and baseline tokens and adjust if needed
                p_len = torch.sum(batch["attention_mask"])
                b_len = torch.sum(baseline_tokens["attention_mask"])

                adj_n = p_len - b_len
                cls_tok = baseline_tokens["input_ids"][0][0]
                sep_tok = baseline_tokens["input_ids"][0][-1]

                # Hacky thing b/c of the way the diagnostic dataset was created (it was originally created just for prepend)
                # So will always need to adjust the tokens for append
                if perturb_type == "append":
                    filler_tokens = torch.full((adj_n+1,), baseline_tokens["input_ids"][0][1]) # skip CLS token
                    filler_attn_mask = torch.full((adj_n+1,), baseline_tokens["attention_mask"][0][1]) 
                    adj_doc = torch.cat((baseline_tokens["input_ids"][0][2:-1], filler_tokens))
                    baseline_tokens["input_ids"] = torch.cat((cls_tok.view(1), adj_doc, sep_tok.view(1)), dim=0).view(1,-1)
                    baseline_tokens["attention_mask"] = torch.cat((baseline_tokens["attention_mask"][0][1:], filler_attn_mask), dim=0).view(1,-1)
                elif perturb_type == "prepend":
                    # But for prepend, we only need to adjust if the lengths are different
                    if p_len != b_len:
                        filler_tokens = torch.full((adj_n,), baseline_tokens["input_ids"][0][1]) # skip CLS token
                        filler_attn_mask = torch.full((adj_n,), baseline_tokens["attention_mask"][0][1]) 
                        adj_doc = torch.cat((filler_tokens, baseline_tokens["input_ids"][0][1:-1]))
                        baseline_tokens["input_ids"] = torch.cat((cls_tok.view(1), adj_doc, sep_tok.view(1)), dim=0).view(1,-1)
                        baseline_tokens["attention_mask"] = torch.cat((filler_attn_mask, baseline_tokens["attention_mask"][0]), dim=0).view(1,-1)
                

                # Get baseline doc embedding
                baseline_outputs = tl_model(
                    baseline_tokens["input_ids"],
                    return_type="embeddings",
                    one_zero_attention_mask=baseline_tokens["attention_mask"],
                )
                baseline_embedding = baseline_outputs[:,0,:].squeeze(0) #.detach().cpu().numpy()

                # Get scores for baseline and perturbed documents
                baseline_score = torch.matmul(q_embedding, baseline_embedding.t())
                perturbed_score = torch.tensor(tfc1_precomputed_scores[(tfc1_precomputed_scores["qid"] == qid) & (tfc1_precomputed_scores["doc_id"] == int(doc_id))]["p_score"].item())


                '''
                Linear function of score diff, calibrated so that it equals 0 when performance is 
                same as on clean input, and 1 when performance is same as on corrupted input.
                '''
                def ranking_metric(patched_doc_embedding, og_score=baseline_score, p_score=perturbed_score):
                    patched_score = torch.matmul(q_embedding, patched_doc_embedding.t())
                    return (patched_score - og_score) / (p_score - og_score)


                # Setup directories to save results
                if not os.path.exists("results"):
                    os.mkdir("results")
                if not os.path.exists(os.path.join("results", perturb_type)):
                    os.mkdir(f"results/{perturb_type}")
                if not os.path.exists("results_head_decomp"):
                    os.mkdir("results_head_decomp")
                if not os.path.exists(os.path.join("results_head_decomp", perturb_type)):
                    os.mkdir(f"results_head_decomp/{perturb_type}")
                if not os.path.exists("results_attn_pattern"):
                    os.mkdir("results_attn_pattern")

                
                # Patch after each layer (residual stream, attention, MLPs)
                if experiment_type == "block":
                    act_patch_block_every = get_act_patch_block_every(
                        tl_model,
                        device,
                        baseline_tokens,
                        perturbed_cache,
                        ranking_metric
                    )
                    detached_block_results = act_patch_block_every.detach().cpu().numpy()
                    np.save("results/{}/{}_{}_block.npy".format(perturb_type, qid, doc_id), detached_block_results)
                    

                # Patch attention heads
                elif experiment_type == "head_all":
                    act_patch_attn_head_out_all_pos = get_act_patch_attn_head_out_all_pos(
                        tl_model,
                        device,
                        baseline_tokens,
                        perturbed_cache,
                        ranking_metric
                    )
                    detached_head_results = act_patch_attn_head_out_all_pos.detach().cpu().numpy()
                    np.save("results/{}/{}_{}_head.npy".format(perturb_type, qid, doc_id), detached_head_results)

                # Patch heads by position
                elif experiment_type == "head_pos":
                    layer_head_list = [(0,9), (1,6), (2,3), (3,8)]
                    act_patch_attn_head_out_by_pos = get_act_patch_attn_head_by_pos(
                        tl_model,
                        device,
                        baseline_tokens,
                        perturbed_cache,
                        ranking_metric,
                        layer_head_list
                    )
                    detached_head_pos_results = act_patch_attn_head_out_by_pos.detach().cpu().numpy()
                    np.save("results_head_decomp/{}/{}_{}_head_by_pos.npy".format(perturb_type, qid, doc_id), detached_head_pos_results)

                elif experiment_type == "head_attn":
                    # Get attention patterns for head
                    attn_heads = [(0,9), (1,6), (2,3), (3,8)]
                    for layer, head in attn_heads:
                        attn_pattern = perturbed_cache["pattern", layer][:,head].mean(0).detach().cpu().numpy()
                        np.save("results_attn_pattern/{}/{}_{}_{}_{}_attn_pattern.npy".format(perturb_type, qid, doc_id, layer, head), attn_pattern)

                
                elif experiment_type == "labels":
                    decoded_tokens = [tokenizer.decode(tok) for tok in batch["input_ids"][0]]
                    labels = ["{} {}".format(tok,i) for i, tok in enumerate(decoded_tokens)]
                    with open("results/{}/{}_{}_labels.txt".format(perturb_type, qid, doc_id), "w") as f:
                        for item in labels:
                            f.write(str(item) + '\n')


            except Exception as e:
                print("ERROR: {} for query {} and document {}".format(e, qid, doc_id))


    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run activation patching with the specific patching experiment and perturbation types.")
    parser.add_argument("experiment_type", type=str, help="What will be patched (e.g., block).")
    parser.add_argument("perturb_type", type=str, help="The perturbation to apply (e.g., append).")
    
    args = parser.parse_args()

    valid_exp_types = {"block", "head_all", "head_pos", "head_attn", "labels"}
    valid_perturb_types = {"append", "prepend"}

    assert args.experiment_type in valid_exp_types, f"Invalid argument: experiment_type. Must be one of {valid_exp_types}."
    assert args.perturb_type in valid_perturb_types, f"Invalid argument: perturb_type. Must be one of {valid_perturb_types}."
    
    _ = run_experiment(args.experiment_type, args.perturb_type)
