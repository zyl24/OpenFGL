import torch
from flcore.base import BaseServer
from flcore.fedstar._utils import init_structure_encoding

class FedStarServer(BaseServer):
    def __init__(self, args, global_data, data_dir, message_pool, device):
        super(FedStarServer, self).__init__(args, global_data, data_dir, message_pool, device)

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