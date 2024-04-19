import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphSAGE(nn.Module):

    def __init__(self, input_dim, hid_dim, output_dim, dropout):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(input_dim, hid_dim)
        self.conv2 = SAGEConv(hid_dim, output_dim)
        self.dropout = dropout
        self.normalized= False

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        embedding = F.dropout(x, p=self.dropout)
        x = self.conv2(embedding, edge_index)
        return embedding, x
