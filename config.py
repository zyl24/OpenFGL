import argparse


supported_scenario = ["fedgraph", "fedsubgraph"]

supported_fedgraph_datasets = [
"AIDS", "BZR", "COLLAB", "COX2", "DD", "DHFR", "ENZYMES", "IMDB-BINARY", "IMDB-MULTI", "MUTAG", "NCI1", "PROTEINS", "PTC_MR",
"Cora", "CiteSeer", "PubMed", "hERG"
]
supported_fedsubgraph_datasets = [
"Cora", "CiteSeer", "PubMed"
]


supported_fedgraph_simulations = ["fedgraph_cross_domain", "fedgraph_label_dirichlet"]
supported_fedsubgraph_simulations = ["fedsubgraph_label_dirichlet", "fedsubgraph_louvain_clustering", "fedsubgraph_metis_clustering"]

supported_fedgraph_task = ["graph_cls", "graph_reg"]
supported_fedsubgraph_task = ["node_cls", "link_pred", "node_clust"]


supported_fl_algorithm = ["fedavg", "fedprox", "scaffold", "moon", "feddc", "fedproto", "fedtgp", "fedpub", "fedstar", "fedgta", "fedtad"]


supported_metrics = ["accuracy", "precision", "f1", "recall"]


supported_evaluation_modes = ["global_model_on_local_data", "global_model_on_global_data", "local_model_on_local_data", "local_model_on_global_data"]

supported_data_processing = ["raw", "random_feature_mask", "link_random_response", "homo_random_injection", "hete_random_injection"]

supported_fedgraph_model = ["gin"]
supported_fedsubgraph_model = ["gcn", "gat", "graphsage", "sgc", "gcn2"]



parser = argparse.ArgumentParser()

# environment settings
parser.add_argument("--use_cuda", type=bool, default=True)
parser.add_argument("--gpuid", type=int, default=0)
parser.add_argument("--seed", type=int, default=2024)

# global dataset settings 
parser.add_argument("--root", type=str, default="/home/ai2/work/dataset")
parser.add_argument("--scenario", type=str, default="fedgraph", choices=supported_scenario)
parser.add_argument("--dataset", type=list, default=["COX2"])
parser.add_argument("--processing", type=str, default="raw", choices=supported_data_processing)
# post_process: 
# random feature mask ratio
parser.add_argument("--feature_mask_prob", type=float, default=0.1)
# dp parameter: epsilon, support 1) random response for link
parser.add_argument("--dp_epsilon", type=float, default=0.)
# homo/hete random injection
parser.add_argument("--homo_injection_ratio", type=float, default=0.)
parser.add_argument("--hete_injection_ratio", type=float, default=0.)

# fl settings
parser.add_argument("--num_clients", type=int, default=10)
parser.add_argument("--num_rounds", type=int, default=100)
parser.add_argument("--fl_algorithm", type=str, default="fedavg", choices=supported_fl_algorithm)
parser.add_argument("--client_frac", type=float, default=1.0)


# simulation settings
parser.add_argument("--simulation_mode", type=str, default="fedgraph_label_dirichlet", choices=supported_fedgraph_simulations + supported_fedsubgraph_simulations)
parser.add_argument("--dirichlet_alpha", type=float, default=10)
parser.add_argument("--dirichlet_try_cnt", type=int, default=100)
parser.add_argument("--least_samples", type=int, default=5)
parser.add_argument("--louvain_resolution", type=float, default=1)
parser.add_argument("--metis_num_coms", type=float, default=100)

# task settings
parser.add_argument("--task", type=str, default="graph_cls", choices=supported_fedgraph_task + supported_fedsubgraph_task)

# training settings
parser.add_argument("--train_val_test", type=str, default="default_split")
parser.add_argument("--num_epochs", type=int, default=3)
parser.add_argument("--dropout", type=float, default=0.5)
parser.add_argument("--lr", type=float, default=1e-2)
parser.add_argument("--optim", type=str, default="adam")
parser.add_argument("--weight_decay", type=float, default=5e-4)
parser.add_argument("--batch_size", type=int, default=128)


# model settings
parser.add_argument("--model", type=list, default=["gin"], choices=supported_fedgraph_model + supported_fedsubgraph_model)
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--hid_dim", type=int, default=64)

# evaluation settings
parser.add_argument("--metrics", type=list, default=["accuracy"])
parser.add_argument("--evaluation_mode", type=str, default="global", choices=supported_evaluation_modes)



args = parser.parse_args()