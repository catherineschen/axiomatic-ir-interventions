# Axiomatic Causal Interventions for Reverse Engineering Relevance Computation in Neural Retrieval Models

By Catherine Chen, Jack Merullo, and Carsten Eickhoff (Brown University, University of Tübingen)

This code corresponds to the paper: __Axiomatic Causal Interventions for Reverse Engineering Relevance Computation in Neural Retrieval Models__, in _Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR ’24)_, July 14–18, 2024, Washington, DC, USA. [Link to paper](https://arxiv.org/abs/2405.02503)

# Setup

This code uses a copy of the [TransformerLens](https://github.com/neelnanda-io/TransformerLens) package, which we make additional changes to support activation patching in a retrieval setting. Changes made to the original TransformerLens package to support activation patching for retrieval (TAS-B) can be found in the following files:

- `components.py`
- `loading_from_pretrained.py`
- `HookedEncoder.py`

Install necessary packages and locally install our modified version of TransformerLens that supports TAS-B:
```
pip install ./TransformerLens
```

# Code

We provide a demo of how we perform activation patching for retrieval in `retrieval_patching_demo.py`. 

```
python retrieval_patching_demo.py
```

## Experiments

To reproduce the experiments in the papers, first [download the data from Google Drive](https://drive.google.com/file/d/1duqXgx2iqPyoom0Nui3nwy33rqjt5pll/view?usp=drive_link). 

To run the patching experiments:

```
python experiment.py EXPERIMENT_TYPE PERTURB_TYPE
```

The patching experiments are currently designed to be run on a single GPU, and depending on the experiment, can take several hours to days to complete. To save time, we would suggest setting up separate jobs for each type of patching experiment.

`EXPERIMENT_TYPE`:
- `block`: patch blocks for each layer over individual token positions (residual stream, attention block, MLP)
- `head_all`: patch individual attention heads over all token positions
- `head_pos`: patch target attention heads over individual token positions
- `head_attn`: get attention head patterns for target heads
- `labels`: get tokenized documents

`PERTURB_TYPE`:
- `append`: target query term added to the end of a document
- `prepend`: target query term added to the beginning of a document


To visualize the results:
```
python plot_results.py EXPERIMENT_TYPE PERTURB_TYPE
```

-----------
Corresponding author: Catherine Chen (catherine_s_chen [at] brown [dot] edu)

If you find this code or any of the ideas in the paper useful, please consider citing:
```
@inproceedings{chen2024axiomatic,
  title={Axiomatic Causal Interventions for Reverse Engineering Relevance Computation in Neural Retrieval Models},
  author={Chen, Catherine and Merullo, Jack and Eickhoff, Carsten},
  booktitle={Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages={1401--1410},
  year={2024}
}
```
