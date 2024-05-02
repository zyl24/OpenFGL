import copy
import torch
import numpy as np
from torch_geometric.utils import to_scipy_sparse_matrix, to_networkx
from torch_geometric.data import Data
from sknetwork.clustering import Louvain
import sys
from sklearn.cluster import KMeans
import pymetis as metis



def get_subgraph_pyg_data(args, global_dataset, node_list):
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
    local_subgraph.num_global_classes = global_dataset.num_classes
    return local_subgraph



def fedgraph_cross_domain(args, global_dataset):
    print("Conducting fedgraph cross domain simulation...")
    local_data = []
    for client_id in range(args.num_clients):
        local_graphs = global_dataset[client_id] # list(InMemoryDataset) -> InMemoryDataset
        local_graphs.num_global_classes = global_dataset[client_id].num_classes
        local_data.append(local_graphs)
    return local_data



def fedgraph_label_dirichlet(args, global_dataset, shuffle=True):
    print("Conducting fedgraph label dirichlet simulation...")
    
    graph_labels = global_dataset.y.numpy()
    num_clients = args.num_clients
    alpha = args.dirichlet_alpha
    unique_labels, label_counts = np.unique(graph_labels, return_counts=True)
    
    
    print(f"num_classes: {len(unique_labels)}")
    print(f"global label distribution: {label_counts}")
       
    min_size = 0
    K = len(unique_labels)
    N = graph_labels.shape[0]

    try_cnt = 0
    while min_size < args.least_samples:
        if try_cnt > args.dirichlet_try_cnt:
            print(f"Client data size does not meet the minimum requirement {args.least_samples}. Try 'args.dirichlet_alpha' larger than {args.dirichlet_alpha} /  'args.try_cnt' larger than {args.try_cnt} / 'args.least_sampes' lower than {args.least_samples}.")
            sys.exit(0)
            
        client_indices = [[] for _ in range(num_clients)]
        for k in range(K):
            idx_k = np.where(graph_labels == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            proportions = np.array([p*(len(idx_j)<N/num_clients) for p,idx_j in zip(proportions,client_indices)])
            proportions = proportions/proportions.sum()
            proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
            client_indices = [idx_j + idx.tolist() for idx_j,idx in zip(client_indices,np.split(idx_k,proportions))]
            min_size = min([len(idx_j) for idx_j in client_indices])
        try_cnt += 1
   
    local_data = []
    client_label_counts = [[0] * K for _ in range(args.num_clients)]
    for client_id in range(args.num_clients):
        for class_i in range(K):
            client_label_counts[client_id][class_i] = (graph_labels[client_indices[client_id]] == class_i).sum()
        
        list.sort(client_indices[client_id])
        
        local_id_to_global_id = {}
        for local_id, global_id in enumerate(client_indices[client_id]):
            local_id_to_global_id[local_id] = global_id
        
        local_graphs = global_dataset.copy(client_indices[client_id]) # InMemoryDataset -> deep-copy subset
        local_graphs.num_global_classes = global_dataset.num_classes
        local_graphs.global_map = local_id_to_global_id
        local_data.append(local_graphs)
    
    print(f"label_counts:\n{np.array(client_label_counts)}")
    return local_data
    
    
    
    
    
def fedsubgraph_label_dirichlet(args, global_dataset, shuffle=True):
    print("Conducting fedsubgraph label dirichlet simulation...")
    node_labels = global_dataset[0].y.numpy()
    num_clients = args.num_clients
    alpha = args.dirichlet_alpha
    unique_labels, label_counts = np.unique(node_labels, return_counts=True)
    
    print(f"num_classes: {len(unique_labels)}")
    print(f"global label distribution: {label_counts}")
       
    min_size = 0
    K = len(unique_labels)
    N = node_labels.shape[0]

    try_cnt = 0
    while min_size < args.least_samples:
        if try_cnt > args.dirichlet_try_cnt:
            print(f"Client data size does not meet the minimum requirement {args.least_samples}. Try 'args.dirichlet_alpha' larger than {args.dirichlet_alpha} /  'args.try_cnt' larger than {args.try_cnt} / 'args.least_sampes' lower than {args.least_samples}.")
            sys.exit(0)
            
        client_indices = [[] for _ in range(num_clients)]
        for k in range(K):
            idx_k = np.where(node_labels == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            proportions = np.array([p*(len(idx_j)<N/num_clients) for p,idx_j in zip(proportions,client_indices)])
            proportions = proportions/proportions.sum()
            proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
            client_indices = [idx_j + idx.tolist() for idx_j,idx in zip(client_indices,np.split(idx_k,proportions))]
            min_size = min([len(idx_j) for idx_j in client_indices])
        try_cnt += 1
   
    
    local_data = []
    client_label_counts = [[0] * K for _ in range(args.num_clients)]
    for client_id in range(args.num_clients):
        for class_i in range(K):
            client_label_counts[client_id][class_i] = (node_labels[client_indices[client_id]] == class_i).sum()
        local_subgraph = get_subgraph_pyg_data(args, global_dataset, client_indices[client_id])
        local_data.append(local_subgraph)
    print(f"label_counts:\n{np.array(client_label_counts)}")
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
        local_subgraph = get_subgraph_pyg_data(args, global_dataset, client_indices[client_id])
        local_data.append(local_subgraph)

    return local_data


def fedsubgraph_metis_clustering(args, global_dataset):
    print("Conducting fedsubgraph metis clustering simulation...")
    graph_nx = to_networkx(global_dataset[0], to_undirected=True)
    communities = {com_id: {"nodes":[], "num_nodes":0, "label_distribution":[0] * global_dataset.num_classes} 
                            for com_id in range(args.metis_num_coms)}
    n_cuts, membership = metis.part_graph(args.metis_num_coms, graph_nx)
    for com_id in range(args.metis_num_coms):
        com_indices = np.where(np.array(membership) == com_id)[0]
        com_indices = list(com_indices)
        communities[com_id]["nodes"] = com_indices
        communities[com_id]["num_nodes"] = len(com_indices)
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
        local_subgraph = get_subgraph_pyg_data(args, global_dataset, client_indices[client_id])
        local_data.append(local_subgraph)
    
    return local_data
    
    



def fedsubgraph_metis(args, global_dataset):
    print("Conducting fedsubgraph metis simulation...")
    assert args.metis_num_coms == args.num_clients, f"args.metis_num_coms should be equal to args.num_clients."
    
    graph_nx = to_networkx(global_dataset[0], to_undirected=True)
    communities = {com_id: {"nodes":[], "num_nodes":0, "label_distribution":[0] * global_dataset.num_classes} 
                            for com_id in range(args.metis_num_coms)}
    
    n_cuts, membership = metis.part_graph(args.metis_num_coms, graph_nx)
    
    for com_id in range(args.metis_num_coms):
        com_indices = np.where(np.array(membership) == com_id)[0]
        com_indices = list(com_indices)
        
    local_data = []
    
    for client_id in range(args.num_clients):
        local_subgraph = get_subgraph_pyg_data(args, global_dataset, com_indices[client_id])
        local_data.append(local_subgraph)
    
    return local_data
    
    

