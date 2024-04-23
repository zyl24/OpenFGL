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




## FGL Studies
Here we present a summary of papers in the FGL field.



| Name | Node Feature | Node Label | Edge Feature | Edge Label | Graph Label | # Graphs | Materials |
| ---- | ------------ | ---------- | ------------ | ---------- | ---------- | -------- | --------- |
|      |              |            |              |            |            |          |           |




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
| FedGT: Federated Node Classification with Scalable Graph Transformer| arXiv| 2024| [[Paper]](https://arxiv.org/abs/2401.15203)|  
|FedGSL: Federated Graph Structure Learning for Local Subgraph Augmentation | ICBD| 2022| [[Paper]](https://ieeexplore.ieee.org/document/10020771) |
| Federated graph semantic and structural learning| IJCAI|2023 | [[Paper]](https://www.ijcai.org/proceedings/2023/0426.pdf) [[Code]](https://github.com/WenkeHuang/FGSSL)|
| FedGL: Federated graph learning framework with global self-supervision| IS | 2024| [[Paper]](https://www.sciencedirect.com/science/article/pii/S002002552301561X) |
| Deep Efficient Private Neighbor Generation for Subgraph Federated Learning| SDM| 2024 | [[Paper]](https://epubs.siam.org/doi/abs/10.1137/1.9781611978032.92)|
| Federated Node Classification over Graphs with Latent Link-type Heterogeneity| WWW|2023 | [[Paper]](https://dl.acm.org/doi/abs/10.1145/3543507.3583471) [[Code]](https://github.com/Oxfordblue7/FedLIT)|
| FedHGN: a federated framework for heterogeneous graph neural networks| IJCAI| 2023 | [[Paper]](https://dl.acm.org/doi/abs/10.24963/ijcai.2023/412) [[Code]](https://github.com/cynricfu/FedHGN)|
|Federated Graph Neural Networks: Overview, Techniques, and Challenges|TNNLS| 2024 |[[Paper]](https://ieeexplore.ieee.org/abstract/document/10428063)|
    
    
</details>


<details>
    <summary> Survey / Library / Benchmarks</summary>
    
| Title | Venue | Year | Materials |
| ----- | ----- | ---- | --------- |
| Federated graph learning--a position paper| arXiv | 2021 | [[Paper]](https://arxiv.org/abs/2105.11099)| 
| Federated graph machine learning: A survey of concepts, techniques, and applications| SIGKDD | 2022 | [[Paper]](https://dl.acm.org/doi/abs/10.1145/3575637.3575644) |
| Federatedscope-gnn: Towards a unified, comprehensive and efficient package for federated graph learning| KDD| 2022 | [[Paper]](https://dl.acm.org/doi/abs/10.1145/3534678.3539112) [[Code]](https://github.com/alibaba/FederatedScope) |

</details>

## FGL Datasets 
Here we categorize various commonly used graph datasets in recent FGL studies



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
