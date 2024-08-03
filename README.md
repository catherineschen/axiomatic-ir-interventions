# Axiomatic Causal Interventions for Reverse Engineering Relevance Computation in Neural Retrieval Models

By Catherine Chen, Jack Merullo, and Carsten Eickhoff (Brown University, University of Tübingen)

This code corresponds to the paper: __Axiomatic Causal Interventions for Reverse Engineering Relevance Computation in Neural Retrieval Models__, in _Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR ’24)_, July 14–18, 2024, Washington, DC, USA. 

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

# Setup

This code uses a copy of the [TransformerLens](https://github.com/neelnanda-io/TransformerLens) package, which we make additional changes to support activation patching in a retrieval setting. Changes made to the original TransformerLens package to support activation patching for retrieval (TAS-B) can be found in the following files:

- `components.py`
- `loading_from_pretrained.py`
- `HookedEncoder.py`

# Code

We provide a demo of how we perform activation patching for retrieval in `retrieval_patching_demo.py`. 

```
python retrieval_patching_demo.py
```


-----------
Corresponding author: Catherine Chen
