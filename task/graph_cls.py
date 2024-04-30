import torch
import torch.nn as nn
from task.base import BaseTask
from utils.basic_utils import extract_floats, idx_to_mask_tensor, mask_tensor_to_idx
from os import path as osp
from utils.metrics import compute_supervised_metrics
import os
import torch
from utils.task_utils import load_graph_cls_default_model
import pickle
from torch_geometric.loader import DataLoader

    

class GraphClsTask(BaseTask):
    def __init__(self, args, client_id, data, data_dir, device):
        super(GraphClsTask, self).__init__(args, client_id, data, data_dir, device)
        self.step_preprocess = None
        
    def train(self):
        self.model.train()
        for _ in range(self.args.num_epochs):
            for data in self.train_dataloader:
                self.optim.zero_grad()
                embedding, logits = self.model.forward(data)
                loss_train = self.loss_fn(embedding, logits, torch.ones_like(data.y).bool())
                loss_train.backward()
                if self.step_preprocess is not None:
                    self.step_preprocess()
                self.optim.step()
            

        
    def evaluate(self, mute=False):
        eval_output = {}
        self.model.eval()
        
        loss_train = 0
        loss_val = 0
        loss_test = 0
        with torch.no_grad():
            for data in self.train_dataloader:
                embedding, logits = self.model.forward(data)
                loss_train += self.loss_fn(embedding, logits, torch.ones_like(data.y).bool())
            for data in self.val_dataloader:
                embedding, logits = self.model.forward(data)
                loss_val = self.loss_fn(embedding, logits, torch.ones_like(data.y).bool())
            for data in self.test_dataloader:
                embedding, logits = self.model.forward(data)
                loss_test = self.loss_fn(embedding, logits, torch.ones_like(data.y).bool())
        
        eval_output["embedding"] = embedding
        eval_output["logits"] = logits
        eval_output["loss_train"] = loss_train / len(self.train_dataloader)
        eval_output["loss_val"]   = loss_val
        eval_output["loss_test"]  = loss_test
        
        
        metric_train = compute_supervised_metrics(metrics=self.args.metrics, logits=logits[self.train_mask], labels=self.data.y[self.train_mask], suffix="train")
        metric_val = compute_supervised_metrics(metrics=self.args.metrics, logits=logits[self.val_mask], labels=self.data.y[self.val_mask], suffix="val")
        metric_test = compute_supervised_metrics(metrics=self.args.metrics, logits=logits[self.test_mask], labels=self.data.y[self.test_mask], suffix="test")
        eval_output = {**eval_output, **metric_train, **metric_val, **metric_test}
        
        info = ""
        for key, val in eval_output.items():
            try:
                info += f"\t{key}: {val:.4f}"
            except:
                continue
            
                   
            
        prefix = f"[client {self.client_id}]" if self.client_id is not None else "[server]"
        print(prefix+info)
        return eval_output
    
    def loss_fn(self, embedding, logits, mask):
        return self.default_loss_fn(logits[mask], self.data.y[mask])
        
    @property
    def default_model(self):            
        return load_graph_cls_default_model(self.args, input_dim=self.num_feats, output_dim=self.num_classes, client_id=self.client_id)
    
    @property
    def default_optim(self):
        if self.args.optim == "adam":
            from torch.optim import Adam
            return Adam
    
    @property
    def num_samples(self):
        return len(self.data)
    
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
        return 0.1, 0.1, 0.8
        
    
        
        
        
    @property
    def train_val_test_path(self):
        return osp.join(self.data_dir, "graph_cls")
    

    def load_train_val_test_split(self):
        if self.client_id is None: # server
            glb_train = []
            glb_val = []
            glb_test = []
            
            for client_id in range(self.args.num_clients):
                glb_train_path = osp.join(self.train_val_test_path, f"glb_train_{client_id}.pkl")
                glb_val_path = osp.join(self.train_val_test_path, f"glb_val_{client_id}.pkl")
                glb_test_path = osp.join(self.train_val_test_path, f"glb_test_{client_id}.pkl")
                
                with open(glb_train_path, 'rb') as file:
                    glb_train_data = pickle.load(file)
                    glb_train += glb_train_data
                    
                with open(glb_val_path, 'rb') as file:
                    glb_val_data = pickle.load(file)
                    glb_val += glb_val_data
                    
                with open(glb_test_path, 'rb') as file:
                    glb_test_data = pickle.load(file)
                    glb_test += glb_test_data
                
            train_mask = idx_to_mask_tensor(glb_train, self.num_samples).bool()
            val_mask = idx_to_mask_tensor(glb_val, self.num_samples).bool()
            test_mask = idx_to_mask_tensor(glb_test, self.num_samples).bool()
            
            self.train_dataloader = DataLoader(self.data[train_mask], batch_size=self.args.batch_size, shuffle=True)
            self.val_dataloader = DataLoader(self.data[val_mask], batch_size=self.args.batch_size, shuffle=False)
            self.test_dataloader = DataLoader(self.data[test_mask], batch_size=self.args.batch_size, shuffle=False)
            

            
            
        else: # client
            train_path = osp.join(self.train_val_test_path, f"train_{self.client_id}.pt")
            val_path = osp.join(self.train_val_test_path, f"val_{self.client_id}.pt")
            test_path = osp.join(self.train_val_test_path, f"test_{self.client_id}.pt")
            
            if osp.exists(train_path) and osp.exists(val_path) and osp.exists(test_path): 
                train_mask = torch.load(train_path)
                val_mask = torch.load(val_path)
                test_mask = torch.load(test_path)
            else:
                train_mask, val_mask, test_mask = self.local_graph_train_val_test_split(self.data, self.args.train_val_test)
                
                if not osp.exists(self.train_val_test_path):
                    os.makedirs(self.train_val_test_path)
                    
                torch.save(train_mask, train_path)
                torch.save(val_mask, val_path)
                torch.save(test_mask, test_path)
                
                # map to global
                glb_train_id = []
                glb_val_id = []
                glb_test_id = []
                for id_train in train_mask.nonzero():
                    glb_train_id.append(self.data.global_map[id_train.item()])
                for id_val in val_mask.nonzero():
                    glb_val_id.append(self.data.global_map[id_val.item()])
                for id_test in test_mask.nonzero():
                    glb_test_id.append(self.data.global_map[id_test.item()])
                with open(osp.join(self.train_val_test_path, f"glb_train_{self.client_id}.pkl"), 'wb') as file:
                    pickle.dump(glb_train_id, file)
                with open(osp.join(self.train_val_test_path, f"glb_val_{self.client_id}.pkl"), 'wb') as file:
                    pickle.dump(glb_val_id, file)
                with open(osp.join(self.train_val_test_path, f"glb_test_{self.client_id}.pkl"), 'wb') as file:
                    pickle.dump(glb_test_id, file)
            
        self.train_mask = train_mask.to(self.device)
        self.val_mask = val_mask.to(self.device)
        self.test_mask = test_mask.to(self.device)
        self.train_dataloader = DataLoader(self.data[self.train_mask], batch_size=self.args.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(self.data[self.val_mask], batch_size=self.args.batch_size, shuffle=False)
        self.test_dataloader = DataLoader(self.data[self.test_mask], batch_size=self.args.batch_size, shuffle=False)
            

    def local_graph_train_val_test_split(self, local_graphs, split):
        num_graphs = self.num_samples
        
        if split == "default_split":
            train_, val_, test_ = self.default_train_val_test_split
        else:
            train_, val_, test_ = extract_floats(split)
        
        train_mask = idx_to_mask_tensor([], num_graphs)
        val_mask = idx_to_mask_tensor([], num_graphs)
        test_mask = idx_to_mask_tensor([], num_graphs)
        for class_i in range(local_graphs.num_classes):
            class_i_graph_mask = local_graphs.y == class_i
            num_class_i_graphs = class_i_graph_mask.sum()
            train_mask += idx_to_mask_tensor(mask_tensor_to_idx(class_i_graph_mask) [:int(train_ * num_class_i_graphs)], num_graphs)
            val_mask += idx_to_mask_tensor(mask_tensor_to_idx(class_i_graph_mask)[int(train_ * num_class_i_graphs) : int((train_+val_) * num_class_i_graphs)], num_graphs)
            test_mask += idx_to_mask_tensor(mask_tensor_to_idx(class_i_graph_mask)[int((train_+val_) * num_class_i_graphs): ], num_graphs)
        
        
        train_mask = train_mask.bool()
        val_mask = val_mask.bool()
        test_mask = test_mask.bool()
        return train_mask, val_mask, test_mask