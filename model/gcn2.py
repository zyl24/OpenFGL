import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCN2Conv

class GCN2(torch.nn.Module):

    def __init__(self, input_dim, hid_dim, output_dim, dropout):
        super(GCN2, self).__init__()
        self.linear1 = nn.Linear(input_dim, hid_dim)
        self.conv1 = GCN2Conv(hid_dim, alpha=0.1)
        self.conv2 = GCN2Conv(hid_dim, alpha=0.1)
        self.linear2 = nn.Linear(hid_dim, output_dim)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.linear1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout)

        x0 = x
        x = self.conv1(x, x0, edge_index)
        embedding = self.conv2(x, x0, edge_index)

        x = self.linear2(embedding)
        return embedding, x