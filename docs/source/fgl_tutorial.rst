Tutorial
========================

Get Started
-----------

You can modify the experimental settings in ``/config.py`` as needed,
and then run ``/main.py`` to start your work with OpenFGL. Moreover, we
provide various configured jupyter notebook examples, all of which can
be found in ``/examples``.

Scenario and Dataset Simulation Settings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   --scenario           # fgl scenario
   --root               # root directory for datasets
   --dataset            # list of used dataset(s)
   --simulation_mode    # strategy for extracting FGL dataset from global dataset
   --processing         # data preprocessing


:scenario:
   "fedgraph", "fedsubgraph"


:simulation_mode:
   *scenario=fedsubgraph*: "fedsubgraph_label_dirichlet", "fedsubgraph_louvain_clustering", "fedsubgraph_metis_clustering", "fedsubgraph_louvain", "fedsubgraph_metis"

   *scenario=fedgraph*: "fedgraph_cross_domain", "fedgraph_label_dirichlet", "fedgraph_topology_skew"


:processing:
   "raw", "random_feature_sparsity", "random_feature_noise", "random_edge_sparsity", "random_edge_noise", "random_label_sparsity", "random_label_noise"

Communication Settings
~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   --num_clients        # number of clients
   --num_rounds         # number of communication rounds
   --client_frac        # client activation fraction


FL/FGL Algorithm Settings
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   --fl_algorithm       # used fl/fgl algorithm


:fl_algorithm:
   *choices*: "isolate", "fedavg", "fedprox", "scaffold", "moon", "feddc", "fedproto", "fedtgp", "fedpub", "fedstar", "fedgta", "fedtad", "gcfl_plus", "fedsage_plus", "adafgl", "feddep", "fggp", "fgssl", "fedgl"


Model and Task Settings
~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   --task               # downstream task
   --train_val_test     # train/validatoin/test split proportion
   --num_epochs         # number of local epochs
   --dropout            # dropout
   --lr                 # learning rate
   --optim              # optimizer
   --weight_decay       # weight decay
   --model              # gnn backbone
   --hid_dim            # number of hidden layer units


:task:
   *scenario=fedgraph*: "graph_cls", "graph_reg"

   *scenario=fedsubgraph*: "node_cls", "link_pred", "node_clust"


Evaluation Settings
~~~~~~~~~~~~~~~~~~~

.. code:: python

   --metrics            # performance evaluation metric
   --evaluation_mode    # personalized evaluation / global evaluation


:metrics:
   *choices*: "accuracy", "precision", "f1", "recall", "auc", "ap", "clustering_accuracy", "nmi", "ari"


:evaluation_mode:
   *choices*: "global_model_on_local_data", "global_model_on_global_data", "local_model_on_local_data", "local_model_on_global_data"


Privacy Settings:
~~~~~~~~~~~~~~~~~~

.. code:: python

   --dp_mech            # differential privacy mechanism
   --dp_eps             # differential privacy epsilon
   --dp_delta           # differential privacy delta
   --grad_clip          # gradient clip max_norm