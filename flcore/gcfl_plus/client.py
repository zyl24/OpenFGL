import torch
import torch.nn as nn
from flcore.base import BaseClient
import numpy as np


class GCFLPlusClient(BaseClient):
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super(GCFLPlusClient, self).__init__(args, client_id, data, data_dir, message_pool, device)
        self.W = {key: value for key, value in self.task.model.named_parameters()}
        self.dW = {key: torch.zeros_like(value) for key, value in self.task.model.named_parameters()}
        self.W_old = {key: value.data.clone() for key, value in self.task.model.named_parameters()}
        self.gconvNames = []
        for k,v in self.task.model.named_parameters():
            if 'convs' in k:
                self.gconvNames.append(k)

    def execute(self):
        for i, ids in enumerate(self.message_pool["server"]["cluster_indices"]):
            if self.client_id in ids:
                j = ids.index(self.client_id)
                tar = self.message_pool["server"]["cluster_weights"][i][j]
                for k in tar:
                    self.W[k].data = tar[k].data.clone()


        for k in self.gconvNames:
            self.W_old[k].data = self.W[k].data.clone()

        self.task.train()

        for k in self.gconvNames:
            self.dW[k].data = self.W[k].data.clone() - self.W_old[k].data.clone()

        # self.weightsNorm = torch.norm(flatten(self.W)).item()
        #
        # weights_conv = {key: self.W[key] for key in self.gconvNames}
        # self.convWeightsNorm = torch.norm(flatten(weights_conv)).item()
        #
        # dWs_conv = {key: self.dW[key] for key in self.gconvNames}
        # self.convDWsNorm = torch.norm(flatten(dWs_conv)).item()

        # grads = {key: value.grad for key, value in self.W.items()}
        # self.gradsNorm = torch.norm(flatten(grads)).item()

        grads_conv = {key: self.W[key].grad for key in self.gconvNames}
        self.convGradsNorm = torch.norm(flatten(grads_conv)).item()

        for k in self.gconvNames:
            self.W[k].data = self.W_old[k].data.clone()

    def send_message(self):
        self.message_pool[f"client_{self.client_id}"] = {
            "num_samples": self.task.num_samples,
            "W": self.W,
            "convGradsNorm": self.convGradsNorm,
            "dW": self.dW
        }

def flatten(w):
    return torch.cat([v.flatten() for v in w.values()])