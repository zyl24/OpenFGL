import copy
import os
import random
import torch
import numpy as np
from torch_geometric.utils import subgraph, to_scipy_sparse_matrix, to_networkx
from torch_geometric.data import Data
from sknetwork.clustering import Louvain
from sklearn.cluster import KMeans
import pymetis as metis
from utils.basic_utils import extract_floats, idx_to_mask_tensor
# from task.fedsubgraph.


def get_subgraph_pyg_data(global_dataset, node_list):
    global_edge_index = global_dataset.edge_index
    node_id_set = set(node_list)
    global_id_to_local_id = {}
    local_id_to_global_id = {}
    local_edge_list = []
    for local_id, global_id in enumerate(node_list):
        global_id_to_local_id[global_id] = local_id
        local_id_to_global_id[local_id] = global_id
        
    for edge_id in range(global_edge_index.shape[1]):
        src = global_edge_index[0, edge_id].item()
        tgt = global_edge_index[1, edge_id].item()
        if src in node_id_set and tgt in node_id_set:
            local_id_src = global_id_to_local_id[src]
            local_id_tgt = global_id_to_local_id[tgt]
            local_edge_list.append((local_id_src, local_id_tgt))
    local_edge_index = torch.tensor(local_edge_list).T
    
    
    local_subgraph = Data(x=global_dataset[0].x[node_list], edge_index=local_edge_index, y=global_dataset[0].y[node_list])
    local_subgraph.global_map = local_id_to_global_id
    local_subgraph.num_classes = global_dataset.num_classes
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



def fedgraph_cross_domain(args, global_dataset):
    print("Conducting fedgraph cross domain simulation...")
    local_data = []
    for client_id in range(args.num_clients):
        local_graphs = global_dataset[client_id]
        local_data.append(local_graphs)
    return local_data



def fedgraph_label_dirichlet(args, global_dataset, shuffle=True):
    print("Conducting fedgraph label dirichlet simulation...")
    num_graphs = len(global_dataset)
    graph_labels = global_dataset.y.numpy()
    num_clients = args.num_clients
    alpha = args.dirichlet_alpha
    unique_labels, label_counts = np.unique(graph_labels, return_counts=True)
    partition_matrix = np.zeros((len(unique_labels), num_clients))
    for i, label in enumerate(unique_labels):
        proportions = np.random.dirichlet(alpha*np.ones(num_clients))
        partition_matrix[i] = np.round(proportions * label_counts[i])
        partition_matrix[i, -1] = label_counts[i] - partition_matrix[i,:-1].sum()
    client_indices = [[] for _ in range(num_clients)]
    for i, label in enumerate(unique_labels):
        label_indices = np.where(graph_labels == label)[0]
        if shuffle:
            np.random.shuffle(label_indices)
        cum_partitions = np.cumsum(partition_matrix[i]).astype(int)
        prev_partition = 0
        for client_id, partition in enumerate(cum_partitions):
            client_indices[client_id].extend(label_indices[prev_partition:partition])
            prev_partition = partition

    local_data = []

    for client_id in range(args.num_clients):
        list.sort(client_indices[client_id])
        
        local_id_to_global_id = {}
        for local_id, global_id in enumerate(client_indices[client_id]):
            local_id_to_global_id[local_id] = global_id
        
        local_graphs = global_dataset[client_indices[client_id]]
        local_graphs.global_map = local_id_to_global_id
        local_data.append(local_graphs)
    
    return local_data
    
    
    
    
    
def fedsubgraph_label_dirichlet(args, global_dataset, shuffle=True):
    print("Conducting fedsubgraph label dirichlet simulation...")
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
            
    local_data = []
    
    for client_id in range(args.num_clients):
        local_subgraph = get_subgraph_pyg_data(global_dataset, client_indices[client_id])
        local_data.append(local_subgraph)

    return local_data


def fedsubgraph_louvain_clustering(args, global_dataset):
    print("Conducting fedsubgraph louvain clustering simulation...")
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

    client_indices = {client_id: [] for client_id in range(args.num_clients)}
    
    for com_id in range(num_communities):
        client_indices[clustering_labels[com_id]] += communities[com_id]["nodes"]
        
    local_data = []
    for client_id in range(args.num_clients):
        local_subgraph = get_subgraph_pyg_data(global_dataset, client_indices[client_id])
        local_data.append(local_subgraph)

    return local_data


def fedsubgraph_metis_clustering(args, global_dataset):
    print("Conducting fedsubgraph metis clustering simulation...")
    graph_nx = to_networkx(global_dataset[0], to_undirected=True)
    communities = {com_id: {"nodes":[], "num_nodes":0, "label_distribution":[0] * global_dataset.num_classes} 
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

    client_indices = {client_id: [] for client_id in range(args.num_clients)}
    
    for com_id in range(num_communities):
        client_indices[clustering_labels[com_id]] += communities[com_id]["nodes"]
    
    local_data = []
    for client_id in range(args.num_clients):
        local_subgraph = get_subgraph_pyg_data(global_dataset, client_indices[client_id])
        local_data.append(local_subgraph)
    
    return local_data
    