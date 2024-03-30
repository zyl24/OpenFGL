import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

INPUT_DIM = 1433
HID_DIM = 64
OUTPUT_DIM = 7
DROPOUT = 0.5
NUM_CLIENTS = 10
NUM_ROUNDS = 100
ACTIVATE_FRAC = 0.5
LR = 1e-2
WEIGHT_DECAY = 5e-4
NUM_EPOCHS = 3


DATASET = "Cora"
PARTITION = "Louvain"


ROOT_PATH = "/home/ubuntu/data/FEDPG/"



def accuracy(logits, label):
    pred = logits.max(1)[1]
    correct = (pred == label).sum()
    total = logits.shape[0]
    return correct / total * 100

class GCN(nn.Module):
    
    def __init__(self, input_dim, hid_dim, output_dim, dropout):
        super(GCN, self).__init__()
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.conv1 = GCNConv(in_channels=self.input_dim, out_channels=self.hid_dim)
        self.conv2 = GCNConv(in_channels=self.hid_dim, out_channels=self.output_dim)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        embedding = F.dropout(x, p=self.dropout)
        x = self.conv2(embedding, edge_index)
        return embedding, x
        
        





class Client:
    def __init__(self, client_id, message_pool):
        self.client_id = client_id
        self.model = GCN(INPUT_DIM, HID_DIM, OUTPUT_DIM, DROPOUT)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        self.loss_fn = nn.CrossEntropyLoss()
        self.message_pool = message_pool
        self.message_pool[f"client_{self.client_id}"] = {}
        self.data = torch.load(os.path.join(ROOT_PATH, f"data{self.client_id}.pt"))
        
        
    
    def execute(self):
        print(f"[client_{client_id}] local train")

        self.model.train()
        for epoch_i in range(NUM_EPOCHS):
            self.optim.zero_grad()
            embedding, logits = self.model.forward(self.data)
            train_loss = self.loss_fn(logits[self.data.train_idx], self.data.y[self.data.train_idx])
            train_loss.backward()
            self.optim.step()
        self.model.eval()

    def receive_message(self):
        print(f"[client_{self.client_id}] receive message from server")
        with torch.no_grad():
            for (local_param, global_param) in zip(self.model.parameters(), self.message_pool["server"]["weight"]):
                local_param.data = global_param


    def send_message(self):
        self.message_pool[f"client_{self.client_id}"] = {
                "num_nodes": 10,
                "weight":self.model.parameters()
            }
        
    def personalized_evaluate(self):
        self.model.eval()
        with torch.no_grad():
            embedding, logits = self.model.forward(self.data)
            val_acc = accuracy(logits[self.data.val_idx], self.data.y[self.data.val_idx])
            test_acc = accuracy(logits[self.data.test_idx], self.data.y[self.data.test_idx])
        return {
            "val_acc": val_acc,
            "test_acc": test_acc,
            "num_nodes": self.data.x.shape[0]
        }
            
        
        
        

class Server:
    def __init__(self, message_pool):
        self.model = GCN(INPUT_DIM, HID_DIM, OUTPUT_DIM, DROPOUT)
        self.message_pool = message_pool
        self.message_pool["server"] = {}

   
    def execute(self):
        self.receive_message()
        with torch.no_grad():
            num_tot_nodes = sum([self.message_pool[f"client_{client_id}"]["num_nodes"] for client_id in self.message_pool[f"sampled_clients"]])
            for it, client_id in enumerate(self.message_pool["sampled_clients"]):
                weight = self.message_pool[f"client_{client_id}"]["num_nodes"] / num_tot_nodes
                
                for (local_param, global_param) in zip(self.message_pool[f"client_{client_id}"]["weight"], self.model.parameters()):
                    if it == 0:
                        global_param.data = weight * local_param
                    else:
                        global_param.data += weight * local_param



    def receive_message(self):
        for client_id in self.message_pool["sampled_clients"]:
            print(f"[server] receive message from client_{client_id}")
        
        
        
    def send_message(self):
        self.message_pool["server"] = {
            "weight": self.model.parameters()
        }
        print("[server] send message")
            



message_pool = {}
clients = [Client(client_id, message_pool) for client_id in range(NUM_CLIENTS)]
server = Server(message_pool)

best_val_acc = 0
best_test_acc = 0
best_round = 0








class FGLTrainer:
    
    def __init__(self, args):
        pass
    
    def train(self, args=self.args):
        for round_id in range(args.num_rounds):
            sampled_clients = random.sample(list(range(NUM_CLIENTS)), int(NUM_CLIENTS * ACTIVATE_FRAC))
            print(f"sampled_clients: {sampled_clients}")
            message_pool["round"] = round_id
            message_pool["sampled_clients"] = sampled_clients
            server.send_message()
            for client_id in sampled_clients:
                clients[client_id].receive_message()
                clients[client_id].execute()
                clients[client_id].send_message()
            server.receive_message()
            server.execute()
            
            # evaluation
            
            global_val_acc = 0
            global_test_acc = 0
            tot_nodes = 0
            
            for client_id in range(NUM_CLIENTS):
                result = clients[client_id].personalized_evaluate()
                val_acc, test_acc, num_nodes = result["val_acc"], result["test_acc"], result["num_nodes"]
                global_val_acc += val_acc * num_nodes
                global_test_acc += test_acc * num_nodes
                tot_nodes += num_nodes
            global_val_acc /= tot_nodes
            global_test_acc /= tot_nodes
            
            if global_val_acc > best_val_acc:
                best_val_acc = global_val_acc
                best_test_acc = global_test_acc
                best_round = round_id
            
            print(f"best_round: {best_round}\tbest_val: {best_val_acc:.2f}\tbest_test: {best_test_acc:.2f}")
            print("-"*50)
                
    
