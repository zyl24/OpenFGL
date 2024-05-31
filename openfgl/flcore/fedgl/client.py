import torch
import torch.nn as nn
from flcore.base import BaseClient
from flcore.fedgl.models import FedGCN
from torch_geometric.utils import to_torch_csr_tensor
from flcore.fedgl.fedgl_config import config


class FedGLClient(BaseClient):
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super(FedGLClient, self).__init__(args, client_id, data, data_dir, message_pool, device)
        self.task.load_custom_model(FedGCN(nfeat=self.task.num_feats,nhid=self.args.hid_dim,
                                           nclass=self.task.num_global_classes,nlayer=self.args.num_layers,dropout=self.args.dropout))
        self.adj = to_torch_csr_tensor(self.task.data.edge_index)
        self.mask = torch.tensor(list(self.task.data.global_map.values())).to(self.device)

    def get_custom_loss_fn(self):
        def custom_loss_fn(embedding, logits, label, mask):
            loss = torch.nn.functional.cross_entropy(logits[mask], label[mask])
            if self.message_pool["round"] != 0 and config['ssl_loss_weight']>0:
                p_g = self.message_pool["server"]["pseudo_labels"][self.client_id]
                p_m = self.message_pool["server"]["pseudo_labels_mask"][self.client_id]
                local_train_mask = self.task.splitted_data['train_mask'].type(torch.int)
                p_m = p_m - local_train_mask
                p_m[p_m < 0] = 0
                if p_m.sum() == 0:
                    index = torch.where(local_train_mask == 0)[0]
                    tmp = torch.randint(0,index.size(0),(1,))
                    p_m[index[tmp]] = 1
                p_m = p_m.type(torch.bool)
                loss_ssl = torch.nn.functional.cross_entropy(logits[p_m], p_g[p_m].type(torch.long))
                loss += config['ssl_loss_weight'] * loss_ssl

            return loss
        return custom_loss_fn

    def execute(self):
        with torch.no_grad():
            for (local_param, global_param) in zip(self.task.model.parameters(), self.message_pool["server"]["weight"]):
                local_param.data.copy_(global_param)

        self.task.loss_fn = self.get_custom_loss_fn()
        if self.message_pool["round"] != 0 and config['pseudo_graph_weight']>0:
            self.task.splitted_data["data"].adj = self.adj + self.message_pool["server"]["whole_adj"][self.client_id].type(torch.float)
        else:
            self.task.splitted_data["data"].adj = self.adj
        self.task.train()


    def send_message(self):
        self.task.model.eval()
        emb,pred = self.task.model(self.task.data)

        self.message_pool[f"client_{self.client_id}"] = {
            "num_samples": self.task.num_samples,
            "weight": list(self.task.model.parameters()),
            "mask": self.mask,
            "embeddings" : emb,
            "preds": pred
        }

