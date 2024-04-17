import torch
import torch.nn as nn
from flcore.base import BaseClient


class FedAvgClient(BaseClient):
    def __init__(self, args, client_id, data, data_dir, message_pool):
        super(FedAvgClient, self).__init__(args, client_id, data, data_dir, message_pool, custom_model=None, custom_loss_fn=None)
        
    
    def execute(self):
        with torch.no_grad():
            for (local_param, global_param) in zip(self.task.model.parameters(), self.message_pool["server"]["weight"]):
                local_param.data = global_param


        self.task.train()

    def send_message(self):
        self.message_pool[f"client_{self.client_id}"] = {
                "num_nodes": self.task.num_nodes,
                "weight": self.task.model.parameters()
            }
        
    def personalized_evaluate(self):
        return self.task.evaluate()
        
            
        