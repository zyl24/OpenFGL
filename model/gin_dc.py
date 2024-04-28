import torch
from torch_geometric.nn import GCNConv, GINConv, global_add_pool
import torch.nn.functional as F


class GIN_dc(torch.nn.Module):
    def __init__(self, input_dim, hid_dim, output_dim, dropout, args):
        super(GIN_dc, self).__init__()
        self.num_layers = args.nlayer
        self.dropout = dropout

        self.pre = torch.nn.Sequential(torch.nn.Linear(input_dim, hid_dim))

        self.embedding_s = torch.nn.Linear(args.n_se, hid_dim)

        self.graph_convs = torch.nn.ModuleList()
        self.nn1 = torch.nn.Sequential(torch.nn.Linear(hid_dim + hid_dim, hid_dim), torch.nn.ReLU(), torch.nn.Linear(hid_dim, hid_dim))
        self.graph_convs.append(GINConv(self.nn1))
        self.graph_convs_s_gcn = torch.nn.ModuleList()
        self.graph_convs_s_gcn.append(GCNConv(hid_dim, hid_dim))

        for l in range(args.nlayer - 1):
            self.nnk = torch.nn.Sequential(torch.nn.Linear(hid_dim + hid_dim, hid_dim), torch.nn.ReLU(), torch.nn.Linear(hid_dim, hid_dim))
            self.graph_convs.append(GINConv(self.nnk))
            self.graph_convs_s_gcn.append(GCNConv(hid_dim, hid_dim))

        self.Whp = torch.nn.Linear(hid_dim + hid_dim, hid_dim)
        self.post = torch.nn.Sequential(torch.nn.Linear(hid_dim, hid_dim), torch.nn.ReLU())
        self.readout = torch.nn.Sequential(torch.nn.Linear(hid_dim, output_dim))

    def forward(self, data):
        x, edge_index, batch, s = data.x, data.edge_index, data.batch, data.stc_enc
        x = self.pre(x)
        s = self.embedding_s(s)
        for i in range(len(self.graph_convs)):
            x = torch.cat((x, s), -1)
            x = self.graph_convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
            s = self.graph_convs_s_gcn[i](s, edge_index)
            s = torch.tanh(s)

        x = self.Whp(torch.cat((x, s), -1))
        x = global_add_pool(x, batch)   #用于图分类任务
        x = self.post(x)
        y = F.dropout(x, self.dropout, training=self.training)
        y = self.readout(y)
        x = F.log_softmax(x, dim=1)  #和nll_loss适配
        return x,y

