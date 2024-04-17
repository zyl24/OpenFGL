import torch
import torch.nn as nn
from task.base import BaseTask
from utils.basic_utils import extract_floats, idx_to_mask_tensor, mask_tensor_to_idx
from os import path as osp
from utils.metrics import compute_supervised_metrics
import os
import torch
from utils.basic_utils import load_node_cls_default_model



    
    
class NodeClsTask(BaseTask):
    def __init__(self, args, client_id, data, data_dir, device, custom_model=None):
        super(NodeClsTask, self).__init__(args, client_id, data, data_dir, custom_model)
    
    def train(self):
        self.model.train()
        for _ in range(self.args.num_epochs):
            self.optim.zero_grad()
            embedding, logits = self.model.forward(self.data)
            if self.custom_loss_fn is None:
                loss_train = self.default_loss_fn(logits[self.train_mask], self.data.y[self.train_mask])
            else:
                loss_train = self.custom_loss_fn(embedding, logits, self.train_mask)
            loss_train.backward()
            self.optim.step()
        
    def evaluate(self):
        eval_output = {}
        self.model.eval()
        with torch.no_grad():
            embedding, logits = self.model.forward(self.data)
            if self.custom_loss_fn is None:
                loss_train = self.default_loss_fn(logits[self.train_mask], self.data.y[self.train_mask])
                loss_val   = self.default_loss_fn(logits[self.val_mask], self.data.y[self.val_mask])
                loss_test  = self.default_loss_fn(logits[self.test_mask], self.data.y[self.test_mask])
            else:
                loss_train = self.custom_loss_fn(embedding, logits, self.train_mask)
                loss_val = self.custom_loss_fn(embedding, logits, self.val_mask)
                loss_test = self.custom_loss_fn(embedding, logits, self.test_mask)

        

        eval_output["loss_train"] = loss_train
        eval_output["loss_val"]   = loss_val
        eval_output["loss_test"]  = loss_test
        
        
        metric_train = compute_supervised_metrics(metrics=self.args.metrics, logits=logits[self.train_mask], labels=self.data.y[self.train_mask], suffix="train")
        metric_val = compute_supervised_metrics(metrics=self.args.metrics, logits=logits[self.val_mask], labels=self.data.y[self.val_mask], suffix="val")
        metric_test = compute_supervised_metrics(metrics=self.args.metrics, logits=logits[self.test_mask], labels=self.data.y[self.test_mask], suffix="test")
        eval_output = {**eval_output, **metric_train, **metric_val, **metric_test}
        
        info = ""
        for key, val in eval_output.items():
            info += f"\t{key}: {val:.4f}"
                   
            
        prefix = f"[client {self.client_id}]" if self.client_id is not None else "[server]"
        print(prefix+info)
        return eval_output
        
    @property
    def default_model(self):            
        return load_node_cls_default_model(self.args, input_dim=self.num_feats, output_dim=self.num_classes, client_id=self.client_id)
    
    @property
    def default_optim(self):
        if self.args.optim == "adam":
            from torch.optim import Adam
            return Adam
    
    @property
    def num_samples(self):
        return self.data.x.shape[0]
    
    @property
    def num_feats(self):
        return self.data.x.shape[1]
    
    @property
    def num_classes(self):
        return self.data.num_classes
    
    @property
    def default_loss_fn(self):
        return nn.CrossEntropyLoss()
    
    @property
    def default_train_val_test_split(self):
        if self.args.dataset[0] == "Cora":
            return 0.2, 0.4, 0.4
        elif self.args.dataset[0] == "CiteSeer":
            return 0.2, 0.4, 0.4
        elif self.args.dataset[0] == "PubMed":
            return 0.2, 0.4, 0.4
        elif self.args.dataset[0] == "Computers":
            return 0.2, 0.4, 0.4
        
        
        
    @property
    def train_val_test_path(self):
        return osp.join(self.data_dir, "node_cls")
    

    def load_train_val_test_split(self):
        if self.client_id is None: # server
            train_list = []
            val_list = []
            test_list = []
            
        else:        
            train_path = osp.join(self.train_val_test_path, f"train_{self.client_id}.pt")
            val_path = osp.join(self.train_val_test_path, f"val_{self.client_id}.pt")
            test_path = osp.join(self.train_val_test_path, f"test_{self.client_id}.pt")
            
            if osp.exists(train_path) and osp.exists(val_path) and osp.exists(test_path): 
                train_mask = torch.load(train_path)
                val_mask = torch.load(val_path)
                test_mask = torch.load(test_path)
            else:
                train_mask, val_mask, test_mask = self.local_subgraph_train_val_test_split(self.data, self.args.train_val_test)
                
                if not osp.exists(self.train_val_test_path):
                    os.makedirs(self.train_val_test_path)
                    
                torch.save(train_mask, train_path)
                torch.save(val_mask, val_path)
                torch.save(test_mask, test_path)
            
            self.train_mask = train_mask.to(self.device)
            self.val_mask = val_mask.to(self.device)
            self.test_mask = test_mask.to(self.device)
            
            
            
    def local_subgraph_train_val_test_split(self, local_subgraph, split):
        num_nodes = local_subgraph.x.shape[0]
        
        if split == "default_split":
            train_, val_, test_ = self.default_train_val_test_split
        else:
            train_, val_, test_ = extract_floats(split)
        
        train_mask = idx_to_mask_tensor([], num_nodes)
        val_mask = idx_to_mask_tensor([], num_nodes)
        test_mask = idx_to_mask_tensor([], num_nodes)
        for class_i in range(local_subgraph.num_classes):
            class_i_node_mask = local_subgraph.y == class_i
            num_class_i_nodes = class_i_node_mask.sum()
            train_mask += idx_to_mask_tensor(mask_tensor_to_idx(class_i_node_mask) [:int(train_ * num_class_i_nodes)], num_nodes)
            val_mask += idx_to_mask_tensor(mask_tensor_to_idx(class_i_node_mask)[int(train_ * num_class_i_nodes) : int((train_+val_) * num_class_i_nodes)], num_nodes)
            test_mask += idx_to_mask_tensor(mask_tensor_to_idx(class_i_node_mask)[int((train_+val_) * num_class_i_nodes): ], num_nodes)
        
        
        train_mask = train_mask.bool()
        val_mask = val_mask.bool()
        test_mask = test_mask.bool()
        return train_mask, val_mask, test_mask