import copy
import os
import random
import torch
import numpy as np
from torch_geometric.utils import subgraph, to_scipy_sparse_matrix, to_networkx
from sknetwork.clustering import Louvain
from sklearn.cluster import KMeans
import metis

def extract_floats(s):
    from decimal import Decimal
    parts = s.split('-')
    train = float(parts[0])
    val = float(parts[1])
    test = float(parts[2])
    assert Decimal(parts[0]) + Decimal(parts[1]) + Decimal(parts[2]) == Decimal(1)
    return train, val, test

def idx_to_mask_tensor(idx_list, length):
    mask = torch.zeros(length)
    mask[idx_list] = 1
    return mask


def local_subgraph_train_val_test_split(local_subgraph, split, num_classes=None):
    num_nodes = local_subgraph.x.shape[0]
    train_, val_, test_ = extract_floats(split)
    
    local_subgraph.train_mask = idx_to_mask_tensor([], num_nodes)
    local_subgraph.val_mask = idx_to_mask_tensor([], num_nodes)
    local_subgraph.test_mask = idx_to_mask_tensor([], num_nodes)
    for class_i in range(num_classes):
        class_i_node_mask = local_subgraph.y == class_i
        num_class_i_nodes = class_i_node_mask.sum()
        local_subgraph.train_mask += idx_to_mask_tensor(class_i_node_mask.nonzero().squeeze().tolist()[:int(train_ * num_class_i_nodes)], num_nodes)
        local_subgraph.val_mask += idx_to_mask_tensor(class_i_node_mask.nonzero().squeeze().tolist()[int(train_ * num_class_i_nodes) : int((train_+val_) * num_class_i_nodes)], num_nodes)
        local_subgraph.test_mask += idx_to_mask_tensor(class_i_node_mask.nonzero().squeeze().tolist()[int((train_+val_) * num_class_i_nodes): ], num_nodes)
    return local_subgraph


def local_graphs_train_val_test_split(local_graphs, split, num_classes=None):
    num_local_graphs = len(local_graphs)
    train_, val_, test_ = extract_floats(split)
    
    if num_classes is not None: # for classification problem
        local_graphs.train_mask = idx_to_mask_tensor([], num_local_graphs)
        local_graphs.val_mask = idx_to_mask_tensor([], num_local_graphs)
        local_graphs.test_mask = idx_to_mask_tensor([], num_local_graphs)
        for class_i in range(num_classes):
            class_i_local_graphs_mask = local_graphs.y == class_i
            num_class_i_local_graphs = class_i_local_graphs_mask.sum()
            local_graphs.train_mask += idx_to_mask_tensor(class_i_local_graphs_mask.nonzero().squeeze().tolist()[:int(train_ * num_class_i_local_graphs)], num_local_graphs)
            local_graphs.val_mask += idx_to_mask_tensor(class_i_local_graphs_mask.nonzero().squeeze().tolist()[int(train_ * num_class_i_local_graphs) : int((train_+val_) * num_class_i_local_graphs)], num_local_graphs)
            local_graphs.test_mask += idx_to_mask_tensor(class_i_local_graphs_mask.nonzero().squeeze().tolist()[int((train_+val_) * num_class_i_local_graphs): ], num_local_graphs)
    else: # for regression problem
        local_graphs.train_mask = idx_to_mask_tensor(range(int(train_ * num_local_graphs)), num_local_graphs)
        local_graphs.val_mask = idx_to_mask_tensor(range(int(train_ * num_local_graphs), int((train_+val_) * num_local_graphs)), num_local_graphs)
        local_graphs.test_mask = idx_to_mask_tensor(range(int((train_+val_) * num_local_graphs), num_local_graphs), num_local_graphs)
    
    assert (local_graphs.train_mask + local_graphs.val_mask + local_graphs.test_mask).sum() == num_local_graphs

    
def fedgraph_simulation(args, global_dataset, shuffle=True):
    if args.simulation_mode == 'fedgraph_noniid':
        assert type(global_dataset) is list and len(global_dataset) == args.num_clients , f"For fed-graph non-iid simulation, the number of clients must be equal to the number of used datasets (args.num_clients={args.num_clients}; used_datasets: {args.dataset})."
        print("Conducting fed-graph noniid simulation across multiple datasets.")
        fmt_list = copy.deepcopy(args.dataset)
        fmt_list = sorted(fmt_list)
        
        save_dir = os.path.join(args.root, "fedgraph", "simulation", f"noniid_{'_'.join(fmt_list)}", f"{args.num_clients}_clients_{args.train_val_test}")
        
        os.makedirs(save_dir, exist_ok=True)
        for client_id in range(args.num_clients):
            local_graphs = global_dataset[client_id]
            num_graphs = len(local_graphs)
            shuffled_graph_idx = list(range(num_graphs))
            if shuffle:
                random.shuffle(shuffled_graph_idx)
            local_graphs_train_val_test_split(local_graphs[shuffled_graph_idx], args.train_val_test, num_classes=local_graphs.num_classes if local_graphs[0].y.dtype is torch.int64 else None)
            torch.save(local_graphs, os.path.join(save_dir, f"client_{client_id}.pt"))
    elif args.simulation_mode == 'fedgraph_iid':
        assert type(global_dataset) is not list, f"For fed-graph iid simulation, only single dataset is supported."
        print("Conducting fed-graph iid simulation within single dataset.")
        save_dir = os.path.join(args.root, "fedgraph", "simulation", args.dataset[0], f"iid_{args.num_clients}_clients_{args.train_val_test}")
        os.makedirs(save_dir, exist_ok=True)
        num_graphs = len(global_dataset)
        shuffled_graph_idx = list(range(num_graphs))
        if shuffle:
            random.shuffle(shuffled_graph_idx)
        for client_id in range(args.num_clients):
            local_graphs = global_dataset[shuffled_graph_idx[num_graphs * client_id // args.num_clients : num_graphs * (client_id+1) // args.num_clients]]
            local_graphs_train_val_test_split(local_graphs, args.train_val_test, num_classes=global_dataset.num_classes if local_graphs[0].y.dtype is torch.int64 else None)
            torch.save(local_graphs, os.path.join(save_dir, f"client_{client_id}.pt"))
        
        
        
        
        
        
def fedsubgraph_homo_simulation(args, global_dataset, shuffle=True):
    if args.simulation_mode == 'fedsubgraph_label_dirichlet':
        assert type(global_dataset) is not list, f"For fed-subgraph label dirichlet simulation, only single dataset is supported."
        node_labels = global_dataset[0].y.numpy()
        num_clients = args.num_clients
        alpha = args.dirichlet_alpha
        unique_labels, label_counts = np.unique(node_labels, return_counts=True)
        partition_matrix = np.zeros((len(unique_labels), num_clients))
        for i, label in enumerate(unique_labels):
            proportions = np.random.dirichlet(alpha*np.ones(num_clients))
            partition_matrix[i] = np.round(proportions * label_counts[i])
            partition_matrix[i, -1] = label_counts[i] - partition_matrix[i,:-1].sum()
        client_indices = [[] for _ in range(num_clients)]
        for i, label in enumerate(unique_labels):
            label_indices = np.where(node_labels == label)[0]
            if shuffle:
                np.random.shuffle(label_indices)
            cum_partitions = np.cumsum(partition_matrix[i]).astype(int)
            prev_partition = 0
            for client_id, partition in enumerate(cum_partitions):
                client_indices[client_id].extend(label_indices[prev_partition:partition])
                prev_partition = partition
                
        save_dir = os.path.join(args.root, "fedsubgraph", "simulation", args.dataset[0], f"label_dirichlet_{args.dirichlet_alpha}_{args.num_clients}_clients_{args.train_val_test}")
        for client_id in range(args.num_clients):
            local_subgraph = subgraph(client_indices[client_id], edge_index=global_dataset[0].edge_index, relabel_nodes=True)
            local_subgraph = local_subgraph_train_val_test_split(local_subgraph)
            torch.save(local_subgraph, os.path.join(save_dir, f"client_{client_id}.pt"))

    elif args.simulation_mode == 'fedsubgraph_louvain_clustering':
        assert type(global_dataset) is not list, f"For fed-subgraph louvain clustering simulation, only single dataset is supported."
        louvain = Louvain(modularity='newman', resolution=args.louvain_resolution, return_aggregate=True) # resolution 越大产生的社区越多, 社区粒度越小
        adj_csr = to_scipy_sparse_matrix(global_dataset[0].edge_index)
        fit_result = louvain.fit_predict(adj_csr)
        communities = {}
        for node_id, com_id in enumerate(fit_result):
            if com_id not in communities:
                communities[com_id] = {"nodes":[], "num_nodes":0, "label_distribution":[0] * global_dataset.num_classes}
            communities[com_id]["nodes"].append(node_id)
            
        for com_id in communities.keys():
            communities[com_id]["num_nodes"] = len(communities[com_id]["nodes"])
            for node in communities[com_id]["nodes"]:
                label = copy.deepcopy(global_dataset[0].y[node])
                communities[com_id]["label_distribution"][label] += 1

        num_communities = len(communities)
        clustering_data = np.zeros(shape=(num_communities, global_dataset.num_classes))
        for com_id in communities.keys():
            for class_i in range(global_dataset.num_classes):
                clustering_data[com_id][class_i] = communities[com_id]["label_distribution"][class_i]
            clustering_data[com_id, :] /= clustering_data[com_id, :].sum()

        kmeans = KMeans(n_clusters=args.num_clients)
        kmeans.fit(clustering_data)

        clustering_labels = kmeans.labels_

        nodes_each_client = {client_id: [] for client_id in range(args.num_clients)}
        
        for com_id in range(num_communities):
            nodes_each_client[clustering_labels[com_id]] += communities[com_id]["nodes"]
            
        save_dir = os.path.join(args.root, "fedsubgraph", "simulation", args.dataset[0], f"louvain_clustering_{args.louvain_resolution}_{args.num_clients}_clients_{args.train_val_test}")
        for client_id in range(args.num_clients):
            local_subgraph = subgraph(nodes_each_client[client_id], edge_index=global_dataset[0].edge_index, relabel_nodes=True)
            local_subgraph = local_subgraph_train_val_test_split(local_subgraph)
            torch.save(local_subgraph, os.path.join(save_dir, f"client_{client_id}.pt"))

    
    
    elif args.simulation_mode == 'fedsubgraph_metis_clustering':
        assert type(global_dataset) is not list, f"For fed-subgraph metis clustering simulation, only single dataset is supported."
        graph_nx = to_networkx(global_dataset[0], to_undirected=True)
        communities[com_id] = {com_id: {"nodes":[], "num_nodes":0, "label_distribution":[0] * global_dataset.num_classes} 
                               for com_id in range(args.metis_num_coms)}
        n_cuts, membership = metis.part_graph(graph_nx, args.metis_num_coms)
        for com_id in range(args.metis_num_coms):
            com_indices = np.where(np.array(membership) == com_id)[0]
            com_indices = list(com_indices)
            communities[com_id]["nodes"] = com_indices
            communities["num_nodes"] = len(com_indices)
            for node in communities[com_id]["nodes"]:
                label = copy.deepcopy(global_dataset[0].y[node])
                communities[com_id]["label_distribution"][label] += 1
        
        num_communities = len(communities)
        clustering_data = np.zeros(shape=(num_communities, global_dataset.num_classes))
        for com_id in communities.keys():
            for class_i in range(global_dataset.num_classes):
                clustering_data[com_id][class_i] = communities[com_id]["label_distribution"][class_i]
            clustering_data[com_id, :] /= clustering_data[com_id, :].sum()

        kmeans = KMeans(n_clusters=args.num_clients)
        kmeans.fit(clustering_data)

        clustering_labels = kmeans.labels_

        nodes_each_client = {client_id: [] for client_id in range(args.num_clients)}
        
        for com_id in range(num_communities):
            nodes_each_client[clustering_labels[com_id]] += communities[com_id]["nodes"]
            
        save_dir = os.path.join(args.root, "fedsubgraph", "simulation", args.dataset[0], f"metis_clustering_{args.louvain_resolution}_{args.num_clients}_clients_{args.train_val_test}")
        for client_id in range(args.num_clients):
            local_subgraph = subgraph(nodes_each_client[client_id], edge_index=global_dataset[0].edge_index, relabel_nodes=True)
            local_subgraph = local_subgraph_train_val_test_split(local_subgraph)
            torch.save(local_subgraph, os.path.join(save_dir, f"client_{client_id}.pt"))
    