import torch
import torch.nn as nn
from flcore.base import BaseClient




class MoonClient(BaseClient):
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super(MoonClient, self).__init__(args, client_id, data, data_dir, message_pool, device, custom_model=None)
        
        self.moon_mu = 1
        self.temperature = 0.5
        self.prev_local_embedding = None
        self.global_embedding = None
        
        
    def get_custom_loss_fn(self):
        def custom_loss_fn(embedding, logits, mask):
            task_loss = self.task.default_loss_fn(logits[mask], self.task.data.y[mask])
            if self.message_pool["round"] != 0:
                sim_global = torch.cosine_similarity(embedding, self.global_embedding, dim=-1).view(-1, 1)
                sim_prev = torch.cosine_similarity(embedding, self.prev_local_embedding, dim=-1).view(-1, 1)
                logits = torch.cat((sim_global, sim_prev), dim=1) / self.temperature
                lbls = torch.zeros(embedding.size(0)).to(self.device).long()
                contrastive_loss = nn.CrossEntropyLoss()(logits ,lbls)
                moon_loss = self.moon_mu * contrastive_loss
                return task_loss + moon_loss
            else:
                return task_loss
        
        return custom_loss_fn    
    
    def execute(self):
        with torch.no_grad():
            for (local_param, global_param) in zip(self.task.model.parameters(), self.message_pool["server"]["weight"]):   
                local_param.data.copy_(global_param)


        self.global_embedding = self.task.evaluate(mute=True)["embedding"].detach()
        
        
        self.task.custom_loss_fn = self.get_custom_loss_fn()
        self.task.train()
        
        self.prev_local_embedding = self.task.evaluate(mute=True)["embedding"].detach()

    def send_message(self):
        self.message_pool[f"client_{self.client_id}"] = {
                "num_samples": self.task.num_samples,
                "weight": list(self.task.model.parameters())
            }
        
    def personalized_evaluate(self):
        return self.task.evaluate()
        
            
        