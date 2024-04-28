import torch
import torch.nn as nn
from flcore.base import BaseClient
from flcore.fedstar._utils import init_structure_encoding


class FedStarClient(BaseClient):
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super(FedStarClient, self).__init__(args, client_id, data, data_dir, message_pool, device)
        if type(self.data) is list:
            init_structure_encoding(args,self.data,args.type_init)
        else:
            init_structure_encoding(args, [self.data], args.type_init)

    def execute(self):
        with torch.no_grad():
            g_w = self.message_pool["server"]["weight"]
            for k,v in self.task.model.state_dict().items():
                if '_s' in k:
                    v.data = g_w[k].data.clone()

        self.task.train()

    def send_message(self):
        self.message_pool[f"client_{self.client_id}"] = {
            "num_samples": self.task.num_samples,
            "weight": self.task.model.state_dict()
        }

    def personalized_evaluate(self):
        return self.task.evaluate()


