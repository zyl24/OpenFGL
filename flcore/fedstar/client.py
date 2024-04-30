import torch
import torch.nn as nn
from flcore.base import BaseClient
from flcore.fedstar._utils import init_structure_encoding
from flcore.fedstar.gin_dc import DecoupledGIN


class FedStarClient(BaseClient):
    def __init__(self, args, client_id, data, data_dir, message_pool, device, n_rw=16, n_dg=16, type_init='rw_dg'):
        super(FedStarClient, self).__init__(args, client_id, data, data_dir, message_pool, device)
        self.n_rw = n_rw
        self.n_dg = n_dg
        self.n_se = n_rw + n_dg
        self.type_init = type_init
        self.task.model = self.get_custom_model()
        
        if type(self.task.data) is list:
            init_structure_encoding(args, self.task.data, self.type_init)
        else:
            init_structure_encoding(args, [self.task.data], self.type_init)

    def get_custom_model(self):
        return DecoupledGIN(input_dim=self.task.num_feats, hid_dim=self.args.hid_dim, output_dim=self.task.num_classes, n_se=self.n_se, num_layers=self.args.num_layers, dropout=self.args.dropout).to(self.device)
    
    def execute(self):
        with torch.no_grad():
            g_w = self.message_pool["server"]["weight"]
            for k,v in self.task.model.state_dict().items():
                if '_s' in k:
                    v.data = g_w[k].data.clone()

        self.task.train()

    def send_message(self):
        self.message_pool[f"client_{self.client_id}"] = {
            "num_samples": self.task.num_samples,
            "weight": self.task.model.state_dict()
        }

