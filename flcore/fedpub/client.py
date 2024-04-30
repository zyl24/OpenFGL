import torch
import torch.nn as nn
from flcore.base import BaseClient
from fedpub.maskedgcn import MaskedGCN


class FedPubClient(BaseClient):
    def __init__(self, args, client_id, data, data_dir, message_pool, device, l1=1e-3, loc_l2=1e-3):
        super(FedPubClient, self).__init__(args, client_id, data, data_dir, message_pool, device)
        self.l1 = l1
        self.loc_l2 = loc_l2
        self.task.model = self.get_custom_model()

    def get_custom_model(self):
        return MaskedGCN(self.args, input_dim=self.task.num_feats, output_dim=self.task.num_classes, client_id=self.client_id)
    
    
    def get_custom_loss_fn(self):
        def custom_loss_fn(embedding, logits, mask):
            loss = torch.nn.functional.cross_entropy(logits[mask], self.task.data.y[mask])
            for name, param in self.task.model.state_dict().items():
                if 'mask' in name:
                    loss += torch.norm(param.float(), 1) * self.l1
                elif 'conv' in name or 'clsif' in name:
                    if self.message_pool['round'] == 0: continue
                    loss += torch.norm(param.float() - self.prev_w[name], 2) * self.loc_l2
            return loss
        return custom_loss_fn


    def execute(self):
        if f'personalized_{self.client_id}' in self.message_pool["server"]:
            weight = self.message_pool["server"][f'personalized_{self.client_id}']
        else:
            weight = self.message_pool["server"]["weight"]

        self.prev_w = weight
        model_state = self.task.model.state_dict()
        for k, v in weight.items():
            if 'running' in k or 'tracked' in k:
                weight[k] = model_state[k]
                continue
            if 'mask' in k or 'pre' in k or 'pos' in k:
                weight[k] = model_state[k]
                continue

        self.task.model.load_state_dict(weight)
        self.task.loss_fn = self.get_custom_loss_fn()
        self.task.train()

    @torch.no_grad()
    def get_functional_embedding(self):
        self.task.model.eval()
        with torch.no_grad():
            proxy_in = self.message_pool['server']['proxy']
            proxy_in = proxy_in.to(self.device)
            proxy_out,_ = self.task.model(proxy_in)
            proxy_out = proxy_out.mean(dim=0)
            proxy_out = proxy_out.clone().detach().cpu().numpy()
        return proxy_out


    def send_message(self):
        self.message_pool[f"client_{self.client_id}"] = {
            "num_samples": self.task.num_samples,
            "weight": self.task.model.state_dict(),
            "functional_embedding" : self.get_functional_embedding()
        }



