import torch
import torch.nn as nn
from flcore.base import BaseClient
from flcore.fedstar._utils import init_structure_encoding
from flcore.fedstar.gin_dc import DecoupledGIN
from torch_geometric.loader import DataLoader
from flcore.fedstar.fedstar_config import config


class FedStarClient(BaseClient):
    def __init__(self, args, client_id, data, data_dir, message_pool, device):

        super(FedStarClient, self).__init__(args, client_id, data, data_dir, message_pool, device)
        self.task.load_custom_model(DecoupledGIN(input_dim=self.task.num_feats, hid_dim=self.args.hid_dim, output_dim=self.task.num_global_classes, n_se=config["n_rw"] + config["n_dg"], num_layers=self.args.num_layers, dropout=self.args.dropout).to(self.device))
        self.data = init_structure_encoding(config["n_rw"], config["n_dg"], self.data, config["type_init"])
        tmp = torch.nonzero(self.task.train_mask, as_tuple=True)[0]
        self.task.train_dataloader = DataLoader([self.data[i] for i in tmp], batch_size=self.args.batch_size, shuffle=False)
        tmp = torch.nonzero(self.task.val_mask, as_tuple=True)[0]
        self.task.val_dataloader = DataLoader([self.data[i] for i in tmp], batch_size=self.args.batch_size, shuffle=False)
        tmp = torch.nonzero(self.task.test_mask, as_tuple=True)[0]
        self.task.test_dataloader = DataLoader([self.data[i] for i in tmp], batch_size=self.args.batch_size, shuffle=False)
        



    
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

