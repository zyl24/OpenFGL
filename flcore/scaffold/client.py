import torch
import torch.nn as nn
from flcore.base import BaseClient
from torch.optim import Optimizer, Adam
import torch
import copy



class ScaffoldClient(BaseClient):
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super(ScaffoldClient, self).__init__(args, client_id, data, data_dir, message_pool, device, custom_model=None)
            
        self.local_control  =  [torch.zeros_like(p.data, requires_grad=False) for p in self.task.model.parameters()]
        
        
    def execute(self):       
        with torch.no_grad():
            for (local_param, global_param) in zip(self.task.model.parameters(), self.message_pool["server"]["weight"]):
                local_param.data.copy_(global_param)

        # self.task.optim = self.get_custom_optim()
        self.task.postprocess = self.postprocess
        self.task.train()    
        
        self.update_local_control()
        
    def postprocess(self):
        for p, local_control, global_control in zip(self.task.model.parameters(), self.local_control, self.message_pool["server"]["global_control"]):
            if p.grad is None:
                continue
            p.grad.data += global_control - local_control
            
            
    def get_custom_optim(self):
        class ScaffoldOptim(self.task.default_optim):
            def __init__(self, params, lr, weight_decay, local_control, global_control):
                super(ScaffoldOptim, self).__init__(params, lr=lr, weight_decay=weight_decay)
                self.local_control = local_control
                self.global_control = global_control
            
            def step(self):
                # for group in self.param_groups:
                #     for p, local_control, global_control in zip(group['params'], self.local_control, self.global_control):
                #         if p.grad is None:
                #             continue
                #         p.grad.data += global_control - local_control
                super(ScaffoldOptim, self).step()
                
                
        # class ScaffoldOptim(Optimizer):
        #     def __init__(self, params, lr, weight_decay, local_control, global_control):
        #         defaults = dict(lr=lr, weight_decay=weight_decay)
        #         super(ScaffoldOptim, self).__init__(params, defaults)
        #         self.local_control = local_control
        #         self.global_control = global_control
            
        #     def step(self):
        #         for group in self.param_groups:
        #             for p, local_control, global_control in zip(group['params'], self.local_control, self.global_control):
        #                 # if p.grad is None:
        #                     # continue
        #                 dp = p.grad.data + global_control.data - local_control.data
        #                 p.data = p.data - dp.data * group['lr']
            
        return ScaffoldOptim(self.task.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay, local_control=self.local_control, global_control=self.message_pool["server"]["global_control"])


    def update_local_control(self):
        with torch.no_grad():
            for it, (local_state, global_state, global_control) in enumerate(zip(self.task.model.parameters(), self.message_pool["server"]["weight"], self.message_pool["server"]["global_control"])):
                self.local_control[it].data = self.local_control[it].data - global_control.data + (global_state.data - local_state.data) / (self.args.num_epochs)
        

    def send_message(self):
        self.message_pool[f"client_{self.client_id}"] = {
                "num_samples": self.task.num_samples,
                "weight": list(self.task.model.parameters()),
                "local_control": self.local_control
            }
        
    def personalized_evaluate(self):
        return self.task.evaluate()
        
            
        