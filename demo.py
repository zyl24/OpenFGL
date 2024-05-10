import numpy as np
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_undirected, remove_self_loops
from torch_geometric.data import Data
from utils.basic_utils import idx_to_mask_tensor

# to_undirected(edge_index, edge_attr, num_nodes, reduce) -> 
# reduce for generate edge_attr
# -> return (result_edge_index, result_edge_attr)  [1,2] -> [1,2], [2,1]
# remove_self_loops(edge_index, edge_attr) -> return (result_edge_index, result_edge_attr)
