import torch
from flcore.base import BaseServer


class FedAvgServer(BaseServer):
    def __init__(self, args, global_data, data_dir, message_pool):
        super(FedAvgServer, self).__init__(args, global_data, data_dir, message_pool, custom_model=None, custom_optim=None, custom_loss_fn=None)

   
    def execute(self):
        with torch.no_grad():
            num_tot_nodes = sum([self.message_pool[f"client_{client_id}"]["num_nodes"] for client_id in self.message_pool[f"sampled_clients"]])
            for it, client_id in enumerate(self.message_pool["sampled_clients"]):
                weight = self.message_pool[f"client_{client_id}"]["num_nodes"] / num_tot_nodes
                
                for (local_param, global_param) in zip(self.message_pool[f"client_{client_id}"]["weight"], self.model.parameters()):
                    if it == 0:
                        global_param.data = weight * local_param
                    else:
                        global_param.data += weight * local_param
        
        
    def send_message(self):
        self.message_pool["server"] = {
            "weight": self.task.model.parameters()
        }