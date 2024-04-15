import torch
import torch.nn as nn
from utils.basic_utils import extract_floats, idx_to_mask_tensor
from os import path as osp
import os




    
    
class NodeClsTask():
    def __init__(self, args, client_id, data, data_dir, model, optim, ):
        self.client_id = client_id
        self.data = data
        self.data_dir = data_dir
        self.args = args
        self.model = model
        self.optim = optim
        self.load_train_val_test_split()
    
    
    def train(self):
        self.model.train()
        for _ in range(self.args.num_epochs):
            self.optim.zero_grad()
            embedding, logits = self.model.forward(self.data)
            
            
            
        self.model.eval()
        
    def evaluate(self):
        self.model.eval()
        
        
        
        
        
        
        
    def loss_fn(self):
        pass
    
    
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
    
        
    @property
    def basic_loss_fn(self):
        return nn.CrossEntropyLoss()
    
    def load_train_val_test_split(self):
        train_path = osp.join(self.train_val_test_path), f"train_{self.client_id}.pt"
        val_path = osp.join(self.train_val_test_path), f"val_{self.client_id}.pt"
        test_path = osp.join(self.train_val_test_path), f"test_{self.client_id}.pt"
        
        if osp.exists(train_path) and osp.exists(val_path) and osp.exists(test_path): 
            train_mask = torch.load(train_path)
            val_mask = torch.load(val_path)
            test_mask = torch.load(test_path)
        else:
            train_mask, val_mask, test_mask = self.local_subgraph_train_val_test_split(self.data, self.args.train_val_test, num_classes=0)
            
            if not osp.exists(self.train_val_test_path):
                os.makedirs(self.train_val_test_path)
                
            torch.save(train_mask, train_path)
            torch.save(val_mask, val_path)
            torch.save(test_mask, test_path)
        
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
            
    def local_subgraph_train_val_test_split(self, local_subgraph, split):
        num_nodes = local_subgraph.x.shape[0]
        
        if split == "default_split":
            train_, val_, test_ = self.default_train_val_test_split()
        else:
            train_, val_, test_ = extract_floats(split)
        
        train_mask = idx_to_mask_tensor([], num_nodes)
        val_mask = idx_to_mask_tensor([], num_nodes)
        test_mask = idx_to_mask_tensor([], num_nodes)
        for class_i in range(local_subgraph.num_classes):
            class_i_node_mask = local_subgraph.y == class_i
            num_class_i_nodes = class_i_node_mask.sum()
            train_mask += idx_to_mask_tensor(class_i_node_mask.nonzero().squeeze().tolist()[:int(train_ * num_class_i_nodes)], num_nodes)
            val_mask += idx_to_mask_tensor(class_i_node_mask.nonzero().squeeze().tolist()[int(train_ * num_class_i_nodes) : int((train_+val_) * num_class_i_nodes)], num_nodes)
            test_mask += idx_to_mask_tensor(class_i_node_mask.nonzero().squeeze().tolist()[int((train_+val_) * num_class_i_nodes): ], num_nodes)
        return train_mask, val_mask, test_mask