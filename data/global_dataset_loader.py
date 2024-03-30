import os.path as osp
from typing import Callable, List, Optional
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.io import fs
import pickle
from torch_geometric.utils import from_scipy_sparse_matrix
import numpy as np
from torch_geometric.utils import coalesce, cumsum, one_hot, remove_self_loops
from typing import Dict, List, Optional, Tuple

def load_global_dataset(root, scenairo, dataset):
    if scenairo == "fedgraph":
        if dataset in ["AIDS",
                            "BZR",
                            "COLLAB", "COX2",
                            "DD", "DHFR",
                            "ENZYMES",
                            "IMDB-BINARY", "IMDB-MULTI",
                            "MUTAG", "NCI1",
                            "PROTEINS", "PTC_MR"]:
            
            from torch_geometric.datasets import TUDataset
            return TUDataset(root=osp.join(root, "fedgraph"), name=dataset, use_node_attr=True, use_edge_attr=True)
        elif dataset in ["hERG"]:
            return hERGDataset(root=osp.join(root, "fedgraph"), use_node_attr=True, use_edge_attr=True)
            
        
        
                
    elif scenairo == "fedsubgraph":
        assert dataset in []


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

