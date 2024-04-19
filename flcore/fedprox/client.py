import torch
import torch.nn as nn
from flcore.base import BaseClient




class FedProxClient(BaseClient):
    def __init__(self, args, client_id, data, data_dir, message_pool, device, fedprox_mu = 1e-3):
        super(FedProxClient, self).__init__(args, client_id, data, data_dir, message_pool, device, custom_model=None)
        self.fedprox_mu = fedprox_mu
        
        
    def get_custom_loss_fn(self):
        def custom_loss_fn(embedding, logits, mask):
            loss_fedprox = 0
            for local_param, global_param in zip(self.task.model.parameters(), self.message_pool["server"]["weight"]):
                loss_fedprox += self.fedprox_mu / 2 * (local_param - global_param).norm(2)**2
            return self.task.default_loss_fn(logits[mask], self.task.data.y[mask]) + loss_fedprox
        
        return custom_loss_fn    
    
    def execute(self):
        with torch.no_grad():
            for (local_param, global_param) in zip(self.task.model.parameters(), self.message_pool["server"]["weight"]):   
                local_param.data.copy_(global_param)


        self.task.custom_loss_fn = self.get_custom_loss_fn()
        self.task.train()

    def send_message(self):
        self.message_pool[f"client_{self.client_id}"] = {
                "num_samples": self.task.num_samples,
                "weight": list(self.task.model.parameters())
            }
        
    def personalized_evaluate(self):
        return self.task.evaluate()
        
            
        