import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv
from torch_geometric.nn.pool import global_add_pool


class GIN(nn.Module):
    def __init__(self, input_dim, hid_dim, output_dim, num_layers=2, dropout=0.5):
        super().__init__()

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(input_dim, 2 * hid_dim),
                nn.BatchNorm1d(2 * hid_dim),
                nn.ReLU(),
                nn.Linear(2 * hid_dim, hid_dim),
            )
            conv = GINConv(mlp, train_eps=True)
            self.convs.append(conv)
            self.batch_norms.append(nn.BatchNorm1d(hid_dim))
            input_dim = hid_dim
        self.lin1 = nn.Linear(hid_dim, hid_dim)
        self.batch_norm1 = nn.BatchNorm1d(hid_dim)
        self.lin2 = nn.Linear(hid_dim, output_dim)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, edge_index)))
            x = F.dropout(x, p=self.dropout, training=self.training)
        embedding = global_add_pool(x, batch)
        x = F.relu(self.batch_norm1(self.lin1(embedding)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        logits = self.lin2(x)
        return embedding, logits
