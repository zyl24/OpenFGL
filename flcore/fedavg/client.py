import torch
import torch.nn as nn
from model.gcn import GCN
from task.fedsubgraph.node_cls import accuracy

class FedAvgClient:
    def __init__(self, client_id, message_pool, args, data):
        self.args = args
        self.client_id = client_id
        self.model = GCN(1433, 64, 7, 0.5)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=1e-2, weight_decay=5e-4)
        self.loss_fn = nn.CrossEntropyLoss()
        self.message_pool = message_pool
        self.message_pool[f"client_{self.client_id}"] = {}
        self.data = data
        
        
    
    def execute(self):
        print(f"[client_{self.client_id}] local train")

        self.model.train()
        for epoch_i in range(self.args.num_epochs):
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
            
        