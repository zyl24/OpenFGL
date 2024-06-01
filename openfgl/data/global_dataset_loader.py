from os import path as osp
from typing import Callable, List, Optional
import torch
import torch_geometric
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.transforms import OneHotDegree
from torch_geometric.io import fs
import pickle
from torch_geometric.utils import from_scipy_sparse_matrix
import numpy as np
from torch_geometric.utils import coalesce, cumsum, remove_self_loops
from typing import Dict, List, Optional, Tuple


def load_global_dataset(root: str, scenario: str, dataset: str):
    """Load a global dataset based on the given scenario and dataset name.

    Args:
        root (str): The root directory where datasets are stored.
        scenario (str): The scenario type, either "graph_fl" or "subgraph_fl".
        dataset (str): The name of the dataset to load.

    Returns:
        Dataset: The loaded dataset object.
    """
    if scenario == "graph_fl":
        if dataset in  ["AIDS",
                        "BZR",
                        "COX2",
                        "DD", "DHFR",
                        "ENZYMES",
                        "MUTAG", "NCI1",
                        "PROTEINS", "PTC_MR"]:
            
            from torch_geometric.datasets import TUDataset
            return TUDataset(root=osp.join(root, "graph_fl"), name=dataset, use_node_attr=True, use_edge_attr=True)
        elif dataset in ["COLLAB", "IMDB-BINARY", "IMDB-MULTI"]:
            from torch_geometric.datasets import TUDataset
            tudataset = TUDataset(root=osp.join(root, "graph_fl"), name=dataset, use_node_attr=True, use_edge_attr=True)
            max_degree = 0
            for data in tudataset:
                deg = torch_geometric.utils.degree(data.edge_index[1], num_nodes=data.num_nodes)
                max_degree = max(max_degree, max(deg).item())
            tudataset.transform = OneHotDegree(int(max_degree))
            return tudataset
        elif dataset in ["hERG"]:
            return hERGDataset(root=osp.join(root, "graph_fl"), use_node_attr=True, use_edge_attr=True)
            
        
        
                
    elif scenario == "subgraph_fl":
        if dataset in ["Cora", "CiteSeer", "PubMed"]:
            from torch_geometric.datasets import Planetoid
            return Planetoid(root=osp.join(root, "subgraph_fl"), name=dataset)
        elif dataset in ["Photo", "Computers"]:
            from torch_geometric.datasets import Amazon
            return Amazon(root=osp.join(root, "subgraph_fl"), name=dataset)
        elif dataset in ["CS", "Physics"]:
            from torch_geometric.datasets import Coauthor
            return Coauthor(root=osp.join(root, "subgraph_fl"), name=dataset)
        elif dataset in ["Chameleon", "Squirrel"]:
            return WikiPages(root=osp.join(root, "subgraph_fl"), name=dataset)
        elif dataset in ["Tolokers", "Roman-empire", "Amazon-ratings", "Questions", "Minesweeper"]:
            from torch_geometric.datasets import HeterophilousGraphDataset
            return HeterophilousGraphDataset(root=osp.join(root, "subgraph_fl"), name=dataset)
        elif dataset in ["Actor"]:
            from torch_geometric.datasets import Actor
            return Actor(root=osp.join(root, "subgraph_fl"))
        elif dataset in ["ogbn-arxiv", "ogbn-products"]:
            from ogb.nodeproppred import PygNodePropPredDataset
            return PygNodePropPredDataset(root=osp.join(root, "subgraph_fl"), name=dataset)
        elif dataset in ["Genius"]:
            from torch_geometric.datasets import LINKXDataset
            return LINKXDataset(root=osp.join(root, "subgraph_fl"), name=dataset)
        elif dataset in ["DBLP", "IMDB", "Freebase", "ACM"]:
            from torch_geometric.datasets import HGBDataset
            return HGBDataset(root=osp.join(root, "subgraph_fl"), name=dataset)
        elif dataset in ["OAG-Venue", "OAG-L1-Field"]:
            pass
        elif dataset in ["OGB-MAG"]:
            from torch_geometric.datasets import OGB_MAG
            return OGB_MAG(root=osp.join(root, "subgraph_fl"), preprocess="metapath2vec")
        

def cat(seq: List[Optional[torch.Tensor]]) -> Optional[torch.Tensor]:
    values = [v for v in seq if v is not None]
    values = [v for v in values if v.numel() > 0]
    values = [v.unsqueeze(-1) if v.dim() == 1 else v for v in values]
    return torch.cat(values, dim=-1) if len(values) > 0 else None


def split(data: Data, batch: torch.Tensor) -> Tuple[Data, Dict[str, torch.Tensor]]:
    node_slice = cumsum(torch.from_numpy(np.bincount(batch)))

    assert data.edge_index is not None
    row, _ = data.edge_index
    edge_slice = cumsum(torch.from_numpy(np.bincount(batch[row])))

    # Edge indices should start at zero for every graph.
    data.edge_index -= node_slice[batch[row]].unsqueeze(0)

    slices = {'edge_index': edge_slice}
    if data.x is not None:
        slices['x'] = node_slice
    else:
        # Imitate `collate` functionality:
        data._num_nodes = torch.bincount(batch).tolist()
        data.num_nodes = batch.numel()
    if data.edge_attr is not None:
        slices['edge_attr'] = edge_slice
    if data.y is not None:
        assert isinstance(data.y, torch.Tensor)
        if data.y.size(0) == batch.size(0):
            slices['y'] = node_slice
        else:
            slices['y'] = torch.arange(0, int(batch[-1]) + 2, dtype=torch.long)

    return data, slices


class hERGDataset(InMemoryDataset):
    url = "https://fedmol.s3-us-west-1.amazonaws.com/datasets/herg/herg.zip"


    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
        use_node_attr: bool = False,
        use_edge_attr: bool = False,
    ) -> None:
        self.name = "hERG"
        super().__init__(root, transform, pre_transform, pre_filter,
                         force_reload=force_reload)

        out = fs.torch_load(self.processed_paths[0])
        if not isinstance(out, tuple) or len(out) < 3:
            raise RuntimeError(
                "The 'data' object was created by an older version of PyG. "
                "If this error occurred while loading an already existing "
                "dataset, remove the 'processed/' directory in the dataset's "
                "root folder and try again.")
        assert len(out) == 3 or len(out) == 4

        if len(out) == 3:  # Backward compatibility.
            data, self.slices, self.sizes = out
            data_cls = Data
        else:
            data, self.slices, self.sizes, data_cls = out

        if not isinstance(data, dict):  # Backward compatibility.
            self.data = data
        else:
            self.data = data_cls.from_dict(data)

        assert isinstance(self._data, Data)
        if self._data.x is not None and not use_node_attr:
            num_node_attributes = self.num_node_attributes
            self._data.x = self._data.x[:, num_node_attributes:]
        if self._data.edge_attr is not None and not use_edge_attr:
            num_edge_attrs = self.num_edge_attributes
            self._data.edge_attr = self._data.edge_attr[:, num_edge_attrs:]

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, "raw")

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, "processed")

    @property
    def num_node_labels(self) -> int:
        return self.sizes['num_node_labels']

    @property
    def num_node_attributes(self) -> int:
        return self.sizes['num_node_attributes']

    @property
    def num_edge_labels(self) -> int:
        return self.sizes['num_edge_labels']

    @property
    def num_edge_attributes(self) -> int:
        return self.sizes['num_edge_attributes']

    @property
    def raw_file_names(self) -> List[str]:
        names = ['adjacency_matrices.pkl', 'edge_feature_matrices.pkl', 'feature_matrices.pkl', 'labels.npy', 'smiles.pkl']
        return names

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self) -> None:
        fs.cp(self.url, self.raw_dir, extract=True)

    def process(self) -> None:
        with open(osp.join(self.raw_dir, "adjacency_matrices.pkl"), 'rb') as file:
            csr_adj_list = pickle.load(file)
        edge_index_list = []
        num_nodes_list = []
        ptr = 0
        for csr_adj in csr_adj_list:
            edge_index_i, _ = from_scipy_sparse_matrix(csr_adj)            
            source = edge_index_i[0, :]
            target = edge_index_i[1, :]
            selected = source <= target
            edge_index_i = edge_index_i[:, selected]            
            edge_index_list.append(edge_index_i + ptr)
            ptr += csr_adj.shape[0]
            num_nodes_list.append(csr_adj.shape[0])
            
        edge_index = torch.hstack(edge_index_list)

        num_graphs = len(num_nodes_list)
        batch = torch.hstack([torch.tensor([graph_i] * num_nodes_list[graph_i]) for graph_i in range(num_graphs)])

        with open(osp.join(self.raw_dir, "feature_matrices.pkl"), 'rb') as file:
            node_feature_list = pickle.load(file)
        node_attribute = torch.vstack([torch.tensor(node_feature_np) for node_feature_np in node_feature_list])  


        node_label = torch.empty((batch.size(0), 0))

        
        with open(osp.join(self.raw_dir, "edge_feature_matrices.pkl"), 'rb') as file:
            edge_feature_list = pickle.load(file)
        edge_attribute = torch.vstack([torch.tensor(edge_feature_np)[:, 2:] for edge_feature_np in edge_feature_list])

        edge_label = torch.empty((edge_index.size(1), 0))


        x = cat([node_attribute, node_label])
        edge_attr = cat([edge_attribute, edge_label])

        graph_feature_np = np.load(osp.join(self.raw_dir, "labels.npy"))
        y = torch.tensor(graph_feature_np).squeeze()


        num_nodes = int(edge_index.max()) + 1 if x is None else x.size(0)
        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        data, slices = split(data, batch)

        sizes = {
            'num_node_attributes': node_attribute.size(-1),
            'num_node_labels': node_label.size(-1),
            'num_edge_attributes': edge_attribute.size(-1),
            'num_edge_labels': edge_label.size(-1),
        }

        self.data = data
        self.slices = slices

        if self.pre_filter is not None or self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]

            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]

            self.data, self.slices = self.collate(data_list)
            self._data_list = None  # Reset cache.

        assert isinstance(self._data, Data)
        fs.torch_save(
            (self._data.to_dict(), self.slices, sizes, self._data.__class__),
            self.processed_paths[0],
        )

    def __repr__(self) -> str:
        return f'{self.name}({len(self)})'


class WikiPages(InMemoryDataset):
    url = "https://data.dgl.ai/dataset"

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        self.name = name # [chameleon, squirrel]

        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        return ["out1_graph_edges.txt", "out1_node_feature_label.txt"]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self) -> None:
        fs.cp(f"{self.url}/{self.name.lower()}.zip", self.raw_dir, extract=True)

    def process(self) -> None:
        edge_index_path = osp.join(self.raw_dir, "out1_graph_edges.txt")
        data_list = []
        with open(edge_index_path, 'r') as file:
            # Skip the header
            next(file)
            for line in file:
                data_list.append([int(number) for number in line.split()])
        edge_index = torch.tensor(data_list).long().T

        node_feature_label_path = osp.join(self.raw_dir, "out1_node_feature_label.txt")
        node_feature_list = []
        node_label_list = []
        with open(node_feature_label_path, 'r') as file:
            # Skip the header
            next(file)
            for line in file:
                node_id, feature, label = line.strip().split('\t')
                node_feature_list.append([int(num) for num in feature.split(',')])
                node_label_list.append(int(label))
        x = torch.tensor(node_feature_list)
        y = torch.tensor(node_label_list)
        data = Data(x=x, edge_index=edge_index, y=y)
        
        
        data = data if self.pre_transform is None else self.pre_transform(data)
        self.save([data], self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name}()'



