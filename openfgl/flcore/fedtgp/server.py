import torch
from flcore.base import BaseServer
import torch.nn as nn
import torch.nn.functional as F

class Trainable_prototypes(nn.Module):
    def __init__(self, num_classes, server_hidden_dim, feature_dim, device):
        super().__init__()

        self.device = device

        self.embedings = nn.Embedding(num_classes, feature_dim)
        layers = [nn.Sequential(
            nn.Linear(feature_dim, server_hidden_dim), 
            nn.ReLU()
        )]
        self.middle = nn.Sequential(*layers)
        self.fc = nn.Linear(server_hidden_dim, feature_dim)

    def forward(self, class_id):
        class_id = torch.tensor(class_id, device=self.device)

        emb = self.embedings(class_id)
        mid = self.middle(emb)
        z = self.fc(mid)
        return z
    
    
class FedTGPServer(BaseServer):
    def __init__(self, args, global_data, data_dir, message_pool, device, fedtgp_lambda=1, num_glb_epochs=10, lr_glb=1e-2):
        super(FedTGPServer, self).__init__(args, global_data, data_dir, message_pool, device)
        
        self.fedtgp_lambda = fedtgp_lambda
        self.num_glb_epochs = num_glb_epochs
        self.lr_glb = lr_glb
        self.trainable_prototypes = Trainable_prototypes(self.task.num_global_classes, args.hid_dim, args.hid_dim, device).to(device)
        self.gp_optimizer = torch.optim.Adam(self.trainable_prototypes.parameters(), lr=lr_glb, weight_decay=args.weight_decay)
        self.global_prototype = {}
        
        
    def execute(self):
        y_list = []
        tensor_list = []
        for client_i in self.message_pool["sampled_clients"]:
            for class_i in range(self.task.num_global_classes):
                y_list.append(class_i)
                tensor_list.append(self.message_pool[f"client_{client_i}"]["local_prototype"][class_i])
        y = torch.tensor(y_list).type(torch.int64).to(self.device)
        all_local_prototypes = torch.cat([v.unsqueeze(0) for v in tensor_list], dim=0)
        row_id = [class_id for class_id in range(self.task.num_global_classes)]
            
        avg_proto = torch.zeros((self.task.num_global_classes, all_local_prototypes.shape[1])).to(self.device)
        num_local_prototypes = len(tensor_list) 
        for proto_i in range(num_local_prototypes):
            avg_proto[y_list[proto_i]] += all_local_prototypes[proto_i,:]
        for class_i in range(self.task.num_global_classes):
            avg_proto /= y_list.count(class_i)
            
                
        gap = torch.ones(self.task.num_global_classes, device=self.device) * 1e9  
            

        for k1 in range(self.task.num_global_classes):
            for k2 in range(self.task.num_global_classes):
                if k1 > k2:
                    dis = torch.norm(avg_proto[k1] - avg_proto[k2], p=2)
                    gap[k1] = torch.min(gap[k1], dis)
                    gap[k2] = torch.min(gap[k2], dis)
        min_gap = torch.min(gap)
        for i in range(len(gap)):
            if gap[i] > torch.tensor(1e8, device=self.device):
                gap[i] = min_gap
        max_gap = torch.max(gap)
        
        for _ in range(self.num_glb_epochs):
            self.gp_optimizer.zero_grad()  
            global_prototypes = self.trainable_prototypes.forward(row_id)
            

            features_square = torch.sum(torch.pow(all_local_prototypes, 2), 1, keepdim=True)
            centers_square = torch.sum(torch.pow(global_prototypes, 2), 1, keepdim=True)
            features_into_centers = torch.matmul(all_local_prototypes, global_prototypes.T)
            dist = features_square - 2 * features_into_centers + centers_square.T
            dist = torch.sqrt(dist)
            
            one_hot = F.one_hot(y, self.task.num_global_classes).to(self.device)
            gap2 = min(max_gap.item(), 100)
            
            
            dist = dist + one_hot * gap2
            glb_loss = 0
            glb_loss = nn.CrossEntropyLoss()(-dist, y)

            glb_loss.backward()
            self.gp_optimizer.step()
        
        for class_i in range(self.task.num_global_classes):
            self.global_prototype[class_i] = self.trainable_prototypes.forward(class_i).detach()
        
    def send_message(self):
        self.message_pool["server"] = {
            "global_prototype": self.global_prototype
        }