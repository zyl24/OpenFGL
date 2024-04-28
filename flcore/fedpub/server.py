import torch
from flcore.base import BaseServer
from collections import defaultdict, OrderedDict
import numpy as np
from scipy.spatial.distance import cosine

def from_networkx(G, group_node_attrs=None, group_edge_attrs=None):
    import networkx as nx
    from torch_geometric.data import Data

    G = G.to_directed() if not nx.is_directed(G) else G

    mapping = dict(zip(G.nodes(), range(G.number_of_nodes())))
    edge_index = torch.empty((2, G.number_of_edges()), dtype=torch.long)
    for i, (src, dst) in enumerate(G.edges()):
        edge_index[0, i] = mapping[src]
        edge_index[1, i] = mapping[dst]

    data = defaultdict(list)

    if G.number_of_nodes() > 0:
        node_attrs = list(next(iter(G.nodes(data=True)))[-1].keys())
    else:
        node_attrs = {}

    if G.number_of_edges() > 0:
        edge_attrs = list(next(iter(G.edges(data=True)))[-1].keys())
    else:
        edge_attrs = {}

    for i, (_, feat_dict) in enumerate(G.nodes(data=True)):
        if set(feat_dict.keys()) != set(node_attrs):
            raise ValueError('Not all nodes contain the same attributes')
        for key, value in feat_dict.items():
            data[str(key)].append(value)

    for i, (_, _, feat_dict) in enumerate(G.edges(data=True)):
        if set(feat_dict.keys()) != set(edge_attrs):
            raise ValueError('Not all edges contain the same attributes')
        for key, value in feat_dict.items():
            key = f'edge_{key}' if key in node_attrs else key
            data[str(key)].append(value)

    for key, value in G.graph.items():
        if key == 'node_default' or key == 'edge_default':
            continue  # Do not load default attributes.
        key = f'graph_{key}' if key in node_attrs else key
        data[str(key)] = value

    for key, value in data.items():
        if isinstance(value, (tuple, list)) and isinstance(value[0], torch.Tensor):
            data[key] = torch.stack(value, dim=0)
        else:
            try:
                data[key] = torch.tensor(value)
            except (ValueError, TypeError, RuntimeError):
                pass

    data['edge_index'] = edge_index.view(2, -1)
    data = Data.from_dict(data)

    if group_node_attrs is all:
        group_node_attrs = list(node_attrs)
    if group_node_attrs is not None:
        xs = []
        for key in group_node_attrs:
            x = data[key]
            x = x.view(-1, 1) if x.dim() <= 1 else x
            xs.append(x)
            del data[key]
        data.x = torch.cat(xs, dim=-1)

    if group_edge_attrs is all:
        group_edge_attrs = list(edge_attrs)
    if group_edge_attrs is not None:
        xs = []
        for key in group_edge_attrs:
            key = f'edge_{key}' if key in node_attrs else key
            x = data[key]
            x = x.view(-1, 1) if x.dim() <= 1 else x
            xs.append(x)
            del data[key]
        data.edge_attr = torch.cat(xs, dim=-1)

    if data.x is None and data.pos is None:
        data.num_nodes = G.number_of_nodes()

    return data





class FedPubServer(BaseServer):
    def __init__(self, args, global_data, data_dir, message_pool, device, agg_norm='exp', norm_scale=10, n_proxy=5):
        super(FedPubServer, self).__init__(args, global_data, data_dir, message_pool, device, custom_model=None)
        self.agg_norm = agg_norm
        self.norm_scale = norm_scale
        self.n_proxy = n_proxy
        
        
        
        self.proxy = self.get_proxy_data(self.task.num_feats)

    def execute(self):
        local_embeddings = []
        local_weights = []
        local_samples = []
        for i in self.message_pool["sampled_clients"]:
            tmp = self.message_pool[f"client_{i}"]
            local_samples.append(tmp['num_samples'])
            local_weights.append(tmp['weight'])
            local_embeddings.append(tmp['functional_embedding'])

        n_connected = len(self.message_pool["sampled_clients"])
        sim_matrix = np.empty(shape=(n_connected, n_connected))
        for i in range(n_connected):
            for j in range(n_connected):
                sim_matrix[i, j] = 1 - cosine(local_embeddings[i], local_embeddings[j])

        if self.agg_norm == 'exp':
            sim_matrix = np.exp(self.norm_scale * sim_matrix)

        row_sums = sim_matrix.sum(axis=1)
        sim_matrix = sim_matrix / row_sums[:, np.newaxis]


        ratio = (np.array(local_samples) / np.sum(local_samples)).tolist()
        self.task.model.load_state_dict(self.aggregate(local_weights, ratio))

        self.update_weights = []
        for i in self.message_pool["sampled_clients"]:
            ratio = sim_matrix[i, :]
            tmp = self.aggregate(local_weights,ratio)
            self.update_weights.append(tmp)



    def send_message(self):
        if self.message_pool["round"] == 0:
            self.message_pool["server"] = {
                "weight": self.task.model.state_dict(),
                "proxy": self.proxy
            }
        else:
            tmp = {}
            for i, id in enumerate(self.message_pool["sampled_clients"]):
                tmp[f'personalized_{id}'] = self.update_weights[i]

            self.message_pool["server"] = {
                "weight": self.task.model.state_dict(),
                "proxy": self.proxy
            }
            self.message_pool["server"].update(tmp)


    def aggregate(self, local_weights, ratio=None):
        aggr_theta = OrderedDict([(k, None) for k in local_weights[0].keys()])
        if ratio is not None:
            for name, params in aggr_theta.items():
                aggr_theta[name] = torch.sum(
                    torch.stack([theta[name] * ratio[j] for j, theta in enumerate(local_weights)]), dim=0)
        else:
            ratio = 1 / len(local_weights)
            for name, params in aggr_theta.items():
                aggr_theta[name] = torch.sum(
                    torch.stack([theta[name] * ratio for j, theta in enumerate(local_weights)]), dim=0)
        return aggr_theta

    def get_proxy_data(self, n_feat):
        import networkx as nx

        num_graphs, num_nodes = self.n_proxy, 100
        data = from_networkx(nx.random_partition_graph([num_nodes] * num_graphs, p_in=0.1, p_out=0, seed=self.args.seed))
        data.x = torch.normal(mean=0, std=1, size=(num_nodes * num_graphs, n_feat))
        return data

