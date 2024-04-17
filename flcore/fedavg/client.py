import torch
import torch.nn as nn
from flcore.base import BaseClient
import copy

class FedAvgClient(BaseClient):
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super(FedAvgClient, self).__init__(args, client_id, data, data_dir, message_pool, device, custom_model=None, custom_loss_fn=None)
            
        
    def execute(self):
        with torch.no_grad():
            for (local_param, global_param) in zip(self.task.model.parameters(), self.message_pool["server"]["weight"]):
                local_param.data = copy.deepcopy(global_param)

        self.task.train()

    def send_message(self):
        self.message_pool[f"client_{self.client_id}"] = {
                "num_samples": self.task.num_samples,
                "weight": list(self.task.model.parameters()) # 需要变为 list 因为迭代器只能遍历一次
            }
        
    def personalized_evaluate(self):
        return self.task.evaluate()
        
            
        