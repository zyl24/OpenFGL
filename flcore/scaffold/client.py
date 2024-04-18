import torch
import torch.nn as nn
from flcore.base import BaseClient
from torch.optim import Optimizer, Adam
import torch
import copy



class ScaffoldClient(BaseClient):
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super(ScaffoldClient, self).__init__(args, client_id, data, data_dir, message_pool, device, custom_model=None)
            
        self.local_control  =  [torch.zeros_like(p, requires_grad=False) for p in self.task.model.parameters()]
        print("ok")

        
    def execute(self):       
        with torch.no_grad():
            for (local_param, global_param) in zip(self.task.model.parameters(), self.message_pool["server"]["weight"]):
                local_param.data.copy_(global_param)

        self.task.postprocess_each_train_epoch = self.postprocess_each_train_epoch
        self.task.train()    
        
        self.update_local_control()
        
    def postprocess_each_train_epoch(self):
        with torch.no_grad():    
            for local_param, local_control, global_control in zip(self.task.model.parameters(), self.local_control, self.message_pool["server"]["global_control"]):
                # local_param.data -= (global_control-local_control) * self.args.lr
                local_param.grad += (global_control-local_control)

    def update_local_control(self):
        with torch.no_grad():
            for it, (local_state, global_state, global_control) in enumerate(zip(self.task.model.parameters(), self.message_pool["server"]["weight"], self.message_pool["server"]["global_control"])):
                self.local_control[it] = self.local_control[it] - global_control + (global_state - local_state) / self.args.num_epochs
        

    def send_message(self):
        self.message_pool[f"client_{self.client_id}"] = {
                "num_samples": self.task.num_samples,
                "weight": list(self.task.model.parameters()),
                "local_control": self.local_control
            }
        
    def personalized_evaluate(self):
        return self.task.evaluate()
        
            
        