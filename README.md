# OpenFGL: A Comprehensive Library and Benchmarks for Federated Graph Learning. 

**OpenFGL** (Open Federated Graph Learning) is a comprehensive, user-friendly algorithm library, complemented by an integrated evaluation platform, designed specifically for researchers in the field of federated graph learning (FGL).



[![Stars](https://img.shields.io/github/stars/zyl24/OpenFGL.svg?color=orange)](https://github.com/zyl24/OpenFGL/stargazers) ![](https://img.shields.io/github/last-commit/zyl24/OpenFGL) [![arXiv](https://img.shields.io/badge/arXiv-2312.04992-b31b1b.svg)](https://arxiv.org/abs/2312.04992)

<!-- [![arXiv](https://img.shields.io/badge/arXiv-2312.04992-b31b1b.svg)](https://arxiv.org/abs/2312.04992) -->
 



## Library Highlights :rocket: 

- Two FGL Scenarios: Fed-Graph and Fed-Subgraph
- xxx FGL Algorithms
- xxx FGL Datasets
- xxx GNN Models
- xxx Downstream Tasks
- Comprehensive FGL Data Property Analysis




## FGL Studies in Top-tier Conferences and Journals
Here we present a summary of papers in the FGL field, featured in top-tier conferences and journals.




<details>
  <summary>Fed-Graph</summary>
    
| Title | Venue | Year | Materials |
| ----- | ----- | ---- | --------- |
| Federated Graph Classification over Non-IID Graphs | NeurIPS  | 2021 | [[Paper]](https://proceedings.neurips.cc/paper/2021/hash/9c6947bd95ae487c81d4e19d3ed8cd6f-Abstract.html) [[Code]](https://github.com/Oxfordblue7/GCFL)  |
|Federated Learning on Non-IID Graphs via Structural Knowledge Sharing| AAAI| 2023| [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/26187) [[Code]](https://github.com/yuetan031/fedstar) |
    


</details>


<details>
  <summary>Fed-Subgraph</summary>
    
| Title | Venue | Year | Materials |
| ----- | ----- | ---- | --------- |
| Subgraph Federated Learning with Missing Neighbor Generation | NeurIPS  | 2021 | [[Paper]](https://proceedings.neurips.cc/paper/2021/hash/34adeb8e3242824038aa65460a47c29e-Abstract.html) [[Code]](https://github.com/zkhku/fedsage)    |
|AdaFGL: A New Paradigm for Federated Node Classification with Topology Heterogeneity| ICDE| 2024 | [[Paper]](https://arxiv.org/abs/2401.11750) [[Code]](https://github.com/xkLi-Allen/AdaFGL) |
|FedGTA: Topology-aware Averaging for Federated Graph Learning | VLDB | 2024| [[Paper]](https://dl.acm.org/doi/abs/10.14778/3617838.3617842) [[Code]](https://github.com/xkLi-Allen/FedGTA)|
|Federated Graph Learning under Domain Shift with Generalizable Prototypes | AAAI | 2024 |[[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/29468) [[Code]](https://github.com/GuanchengWan/FGGP) | 

    
    
</details>








## Get Started
You can modify the experimental settings in `/config.py` as needed, and then run `/main.py` to start your work with OpenFGL. Moreover, we provide various configured jupyter notebook examples, all of which can be found in `/examples`.

### Scenario and Dataset Simulation Settings

```python
--scenario           # fgl scenario
--root               # root directory for datasets
--dataset            # list of used dataset(s)
--simulation_mode    # strategy for extracting FGL dataset from global dataset
```

### Communication Settings

```python
--num_clients        # number of clients
--num_rounds         # number of communication rounds
--client_frac        # client activation fraction
```

### FL/FGL Algorithm Settings
```python
--fl_algorithm       # used fl/fgl algorithm
```

### Model and Task Settings
```python
--task               # downstream task
--train_val_test     # train/validatoin/test split proportion
--num_epochs         # number of local epochs
--dropout            # dropout
--lr                 # learning rate
--optim              # optimizer
--weight_decay       # weight decay
--model              # gnn backbone
--hid_dim            # number of hidden layer units
```


### Evaluation Settings

```python
--metrics            # performance evaluation metric
--evaluation_mode    # personalized evaluation / global evaluation
```
## Cite
Please cite our paper (and the respective papers of the methods used) if you use this code in your own work:
```
@article{xkLi_FedGTA_VLDB_2024,
  author       = {Xunkai Li and
                  Zhengyu Wu and
                  Wentao Zhang and
                  Yinlin Zhu and
                  Ronghua Li and
                  Guoren Wang},
  title        = {FedGTA: Topology-aware Averaging for Federated Graph Learning},
  journal      = {Proc. {VLDB} Endow.},
  volume       = {17},
  number       = {1},
  pages        = {41--50},
  year         = {2023},
  url          = {https://www.vldb.org/pvldb/vol17/p41-li.pdf},
  timestamp    = {Mon, 27 Nov 2023 13:13:34 +0100},
  biburl       = {https://dblp.org/rec/journals/pvldb/LiWZZLW23.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
