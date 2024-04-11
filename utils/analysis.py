import networkx as nx
import torch_geometric
import torch_geometric.data
from torch_geometric.utils import to_networkx
import numpy as np


def average_kl_divergence(label_distributions):
    global_dist = np.mean(label_distributions, axis=0)
    kl_div = np.sum(label_distributions * np.log((label_distributions + 1e-9) / (global_dist + 1e-9)), axis=1)
    return np.mean(kl_div)



def degree_kurtosis(data: torch_geometric.data.Data):
    graph_nx = to_networkx(data, to_undirected=True)
    degrees = [degree for node, degree in graph_nx.degree()]
    kurtosis = np.mean((degrees - np.mean(degrees))**4) / (np.std(degrees)**4) - 3
    return kurtosis


def clustering_coefficient(data: torch_geometric.data.Data):
    graph_nx = to_networkx(data, to_undirected=True)
    clustering_coeffs = nx.clustering(graph_nx)
    return clustering_coeffs

def avg_clustering_coefficient(data: torch_geometric.data.Data):
    graph_nx = to_networkx(data, to_undirected=True)
    avg_clustering_coeffs = nx.average_clustering(graph_nx)
    return avg_clustering_coeffs    

def avg_shortest_path_length(data: torch_geometric.data.Data):
    graph_nx = to_networkx(data, to_undirected=True)

    if nx.is_connected(graph_nx):
        avg_shortest_path_length = nx.average_shortest_path_length(graph_nx)
    else:
        largest_cc = max(nx.connected_components(graph_nx), key=len)
        subgraph = graph_nx.subgraph(largest_cc)
        avg_shortest_path_length = nx.average_shortest_path_length(subgraph)
    
    return avg_shortest_path_length


def largest_component_percentage(data: torch_geometric.data.Data):
    graph_nx = to_networkx(data, to_undirected=True)
    
    largest_cc_size = len(max(nx.connected_components(graph_nx), key=len))
    largest_component_percentage = (largest_cc_size / len(graph_nx.nodes())) * 100

    return largest_component_percentage