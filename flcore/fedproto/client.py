import torch
import torch.nn as nn
from flcore.base import BaseClient

class FedProtoClient(BaseClient):
    def __init__(self, args, client_id, data, data_dir, message_pool, device, fedproto_lambda=1):
        super(FedProtoClient, self).__init__(args, client_id, data, data_dir, message_pool, device)
        self.fedproto_lambda = fedproto_lambda
        self.local_prototype = {}
    
    
    def execute(self):
        self.task.custom_loss_fn = self.get_custom_loss_fn()
        self.task.train()
        self.update_local_prototype()


    def get_custom_loss_fn(self):
        def custom_loss_fn(embedding, logits, mask):
            if self.message_pool["round"] == 0:
                return self.task.default_loss_fn(logits[mask], self.task.data.y[mask]) 
            else:
                loss_fedproto = 0
                for class_i in range(self.task.data.num_classes):
                    selected_idx = self.task.train_mask & (self.task.data.y == class_i)
                    if selected_idx.sum() == 0:
                        continue
                    input = embedding[selected_idx]
                    target = self.message_pool["server"]["global_prototype"][class_i].expand_as(input)
                    loss_fedproto += nn.MSELoss()(input, target)
                return self.task.default_loss_fn(logits[mask], self.task.data.y[mask]) + self.fedproto_lambda * loss_fedproto 
        return custom_loss_fn    
    
    
    def update_local_prototype(self):
        with torch.no_grad():
            embedding = self.task.evaluate(mute=True)["embedding"]
            for class_i in range(self.task.data.num_classes):
                selected_idx = self.task.train_mask & (self.task.data.y == class_i)
                if selected_idx.sum() == 0:
                    self.local_prototype[class_i] = torch.zeros(self.args.hid_dim).to(self.device)
                else:
                    input = embedding[selected_idx]
                    self.local_prototype[class_i] = torch.mean(input, dim=0)
  
            
    def send_message(self):
        self.message_pool[f"client_{self.client_id}"] = {
                "num_samples": self.task.num_samples,
                "local_prototype": self.local_prototype
            }
        
    def personalized_evaluate(self):
        return self.task.evaluate()
        
            
        