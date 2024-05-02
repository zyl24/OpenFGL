import torch
from flcore.base import BaseServer
from flcore.fedstar._utils import init_structure_encoding
from flcore.fedstar.gin_dc import DecoupledGIN
from flcore.fedstar.fedstar_config import config


class FedStarServer(BaseServer):
    def __init__(self, args, global_data, data_dir, message_pool, device):
        super(FedStarServer, self).__init__(args, global_data, data_dir, message_pool, device)
        self.task.load_custom_model(DecoupledGIN(input_dim=self.task.num_feats, hid_dim=self.args.hid_dim, output_dim=self.task.num_global_classes, n_se=config["n_rw"] + config["n_dg"], num_layers=self.args.num_layers, dropout=self.args.dropout).to(self.device))


    def execute(self):
        with torch.no_grad():
            num_tot_samples = sum([self.message_pool[f"client_{client_id}"]["num_samples"] for client_id in
                                   self.message_pool[f"sampled_clients"]])
            for it, client_id in enumerate(self.message_pool["sampled_clients"]):
                weight = self.message_pool[f"client_{client_id}"]["num_samples"] / num_tot_samples
                local_weight = self.message_pool[f"client_{client_id}"]["weight"]

                for k,v in self.task.model.state_dict().items():
                    if '_s' in k:
                        if it == 0:
                            v.data.copy_(weight * local_weight[k])
                        else:
                            v.data += weight * local_weight[k]



    def send_message(self):
        self.message_pool["server"] = {
            "weight": self.task.model.state_dict()
        }