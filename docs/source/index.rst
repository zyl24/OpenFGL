.. OpenFGL documentation master file, created by
   sphinx-quickstart on Fri May 31 17:37:50 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to OpenFGL's documentation!
===================================

.. figure:: img/openfgl_logo.png
   :width: 600
   :align: center


**OpenFGL** is a benchmark for Federated Graph Learning.
It provides a fair and comprehensive platform to evaluate existing FGL works and facilitate future FGL research.


.. note::

   This project is under active development.

Citation
--------
Please cite our paper (and the respective papers of the methods used) if
you use this code in your own work:

::

   @article{li2023fedgta,
     title={FedGTA: Topology-Aware Averaging for Federated Graph Learning},
     author={Li, Xunkai and Wu, Zhengyu and Zhang, Wentao and Zhu, Yinlin and Li, Rong-Hua and Wang, Guoren},
     journal={Proceedings of the VLDB Endowment},
     volume={17},
     number={1},
     pages={41--50},
     year={2023},
     publisher={VLDB Endowment}
   }

   @article{li2024adafgl,
     title={AdaFGL: A New Paradigm for Federated Node Classification with Topology Heterogeneity},
     author={Li, Xunkai and Wu, Zhengyu and Zhang, Wentao and Sun, Henan and Li, Rong-Hua and Wang, Guoren},
     journal={arXiv preprint arXiv:2401.11750},
     year={2024}
   }

   @article{zhu2024fedtad,
     title={FedTAD: Topology-aware Data-free Knowledge Distillation for Subgraph Federated Learning},
     author={Zhu, Yinlin and Li, Xunkai and Wu, Zhengyu and Wu, Di and Hu, Miao and Li, Rong-Hua},
     journal={arXiv preprint arXiv:2404.14061},
     year={2024}
   }



If you use our benchmark in your works, we would appreciate citations to the paper:

.. toctree::
   :glob:
   :maxdepth: 3
   :caption: Getting Started

   installation
   example
   fgl_tutorial


.. toctree::
   :glob:
   :maxdepth: 3
   :caption: Modules

   module_client
   module_server
   module_task


.. toctree::
   :glob:
   :maxdepth: 3
   :caption: Resources

   fgl_studies

