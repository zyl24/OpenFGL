import torch
import torch.nn as nn
from flcore.base import BaseClient
from flcore.fedsage_plus.fedsage_plus_config import config
import numpy as np
from flcore.fedsage_plus.locsage_plus import LocSAGEPlus
from flcore.fedsage_plus._utils import greedy_loss
from data.simulation import get_subgraph_pyg_data
import torch.nn.functional as F
import copy



class FedSagePlusClient(BaseClient):
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super(FedSagePlusClient, self).__init__(args, client_id, data, data_dir, message_pool, device)
        self.task.load_custom_model(LocSAGEPlus(input_dim=self.task.num_feats, 
                                                hid_dim=self.args.hid_dim, 
                                                latent_dim=config["latent_dim"], 
                                                output_dim=self.task.num_global_classes, 
                                                max_pred=config["max_pred"], 
                                                dropout=self.args.dropout))
        
        self.splitted_impaired_data, self.num_missing, self.missing_feat, \
        self.original_neighbors, self.impaired_neighbors = self.get_impaired_subgraph()
                
        
        self.phase = ""
        
        # fill_nodes, fill_G = mending_graph.fill_graph(local_owner.hasG_hide,
        #                                               local_owner.subG,
        #                                               pred_missing, pred_feats, local_owner.feat_shape)

        
    def switch_phase(self, phase):
        assert phase in ["train_fedGen", "train_fedSagePC"]
 

            
            
            
            
    def get_custom_loss_fn(self):
        assert self.phase in ["train_fedGen", "train_fedSagePC"]
        if self.phase == "train_neigh_gen":
            def custom_loss_fn(embedding, logits, label, mask):    
                pred_degree = self.task.model.output_pred_degree.squeeze()
                pred_neig_feat = self.task.model.output_pred_neig_feat
                num_impaired_nodes = self.splitted_impaired_data["data"].x.shape[0]
                impaired_logits = logits[: num_impaired_nodes]
                
                
                loss_train_missing = F.smooth_l1_loss(pred_degree[mask].float(), self.num_missing[mask])

                loss_train_feat = greedy_loss(pred_neig_feat[mask], self.missing_feat[mask], pred_degree[mask], self.num_missing[mask])

                true_nc_label= self.impaired_subgraph.y[mask]
                loss_train_label= F.cross_entropy(impaired_logits[mask], true_nc_label)
                
                loss = (config["num_missing_trade_off"] * loss_train_missing + config["missing_feat_trade_off"] * loss_train_feat + config["cls_trade_off"] * loss_train_label).float()
                
                
                for client_id in self.message_pool["sampled_clients"]:
                    if client_id != self.client_id:
                        others_ids = np.random.choice(self.message_pool[f"client_{client_id}"]["num_samples"], self.task.train_mask.sum())
                        global_target_feat = []
                        for other_central_id in others_ids:
                            other_neighbors = self.message_pool[f"client_{client_id}"]["original_neighbors"][other_central_id]
                            while len(other_neighbors) == 0:
                                other_central_id = np.random.choice(self.message_pool[f"client_{client_id}"]["num_samples"],1)
                                other_neighbors = self.message_pool[f"client_{client_id}"]["original_neighbors"][other_central_id]
                            choice_i = np.random.choice(other_neighbors, config["max_pred"])
                            for ch_i in choice_i:
                                global_target_feat.append(self.message_pool[f"client_{client_id}"]["feat"]([ch_i])[0])
                        global_target_feat = np.asarray(global_target_feat).reshape((self.task.train_mask.sum(), config["max_pred"], self.task.num_feats))
                        loss_train_feat_other = greedy_loss(pred_neig_feat[mask],
                                                                global_target_feat,
                                                                pred_degree[mask],
                                                                self.num_missing[mask]).unsqueeze(0).mean().float()
                        loss += config["missing_feat_trade_off"]  * loss_train_feat_other    
                return loss
            
            
            
        elif self.phase == "train_fedpc":
            pass
        else:
            raise ValueError
        return custom_loss_fn
    
    
    
        
    def get_impaired_subgraph(self):
        hide_len = int(config["hidden_portion"] * (self.task.val_mask).sum())
        could_hide_ids = self.task.val_mask.nonzero().squeeze().tolist()
        hide_ids = np.random.choice(could_hide_ids, hide_len, replace=False)
        all_ids = list(range(self.task.num_samples))
        remained_ids = list(set(all_ids) - set(hide_ids))
        
        impaired_subgraph = get_subgraph_pyg_data(global_dataset=self.task.data, node_list=remained_ids)
        
        num_missing_list = []
        missing_feat_list = []
        
        
        original_neighbors = {node_id: set() for node_id in range(self.task.data.x.shape[0])}
        for edge_id in range(self.task.data.edge_index.shape[1]):
            source = self.task.data.edge_index[0, edge_id].item()
            target = self.task.data.edge_index[1, edge_id].item()
            if source != target:
                original_neighbors[source].add(target)
                original_neighbors[target].add(source)
        
        impaired_neighbors = {node_id: set() for node_id in range(impaired_subgraph.x.shape[0])}
        for edge_id in range(impaired_subgraph.edge_index.shape[1]):
            source = impaired_subgraph.edge_index[0, edge_id].item()
            target = impaired_subgraph.edge_index[1, edge_id].item()
            if source != target:
                original_neighbors[source].add(target)
                original_neighbors[target].add(source)
                
                
        for impaired_id in range(impaired_subgraph.x.shape[0]):
            original_id = impaired_subgraph.global_map[impaired_id]
            num_original_neighbor = len(original_neighbors[original_id])
            num_impaired_neighbor = len(impaired_neighbors[impaired_id])
            impaired_neighbor_in_original = set()
            for impaired_neighbor in impaired_neighbors[impaired_id]:
                impaired_neighbor_in_original.add(impaired_subgraph.global_map[impaired_neighbor])
            
            num_missing_neighbors = num_original_neighbor - num_impaired_neighbor
            num_missing_list.append(num_missing_neighbors)
            missing_neighbors = original_neighbors[original_id] - impaired_neighbor_in_original
            
            
            
            if num_missing_neighbors == 0:
                current_missing_feat = torch.zeros((1, config["max_pred"], self.task.num_feats)).to(self.device)
            else:
                if num_missing_neighbors <= config["max_pred"]:
                    zeros = torch.zeros((max(0, config["max_pred"] - num_missing_neighbors), self.task.num_feats)).to(self.device)
                    current_missing_feat = torch.vstack((self.task.data.x[list(missing_neighbors)], zeros)).view(config["max_pred"], self.task.num_feats)
                else:
                    current_missing_feat = self.task.data.x[list(missing_neighbors)[:config["max_pred"]]].view(config["max_pred"], self.task.num_feats)
            
            missing_feat_list.append(current_missing_feat)
        
        # num_missing  = np.asarray(num_missing_list).view(-1,1)
        # missing_feat = np.asarray(missing_feat_list).view(-1, config["max_pred"], self.task.num_feats)
        
        num_missing = torch.tensor(num_missing_list).squeeze().float().to(self.device)
        missing_feat = torch.stack(missing_feat_list, 0)

        impaired_train_mask = torch.zeros(impaired_subgraph.x.shape[0]).bool().to(self.device)
        impaired_val_mask = torch.zeros(impaired_subgraph.x.shape[0]).bool().to(self.device)
        impaired_test_mask = torch.zeros(impaired_subgraph.x.shape[0]).bool().to(self.device)
        
        for impaired_id in range(impaired_subgraph.x.shape[0]):
            original_id = impaired_subgraph.global_map[impaired_id]
            
            if self.task.train_mask[original_id]:
                impaired_train_mask[impaired_id] = 1
            
            if self.task.val_mask[original_id]:
                impaired_val_mask[impaired_id] = 1
                
            if self.task.test_mask[original_id]:
                impaired_test_mask[impaired_id] = 1
        
        splitted_impaired_data = {
            "data": impaired_subgraph,
            "train_mask": impaired_train_mask,
            "val_mask": impaired_val_mask,
            "test_mask": impaired_test_mask
        }
        
        return splitted_impaired_data, num_missing, missing_feat, original_neighbors, impaired_neighbors
    
    
    
    def local_pretrain(self):
        
        pass
        
    def execute(self):
        with torch.no_grad():
            for (local_param, global_param) in zip(self.task.model.parameters(), self.message_pool["server"]["weight"]):
                local_param.data.copy_(global_param)

        self.task.loss_fn = self.get_custom_loss_fn()
        self.task.train()

    def send_message(self):
        self.message_pool[f"client_{self.client_id}"] = {
                "num_samples": self.task.num_samples,
                "weight": list(self.task.model.parameters()),
                "feat": self.impaired_subgraph.x if self.phase == "train_fedGen" else self.task.data.x,
                "original_neighbors": self.original_neighbors
            }
        
