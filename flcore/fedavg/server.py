import torch
from model.gcn import GCN


class FedAvgServer:
    def __init__(self, message_pool):
        self.model = GCN(1433, 64, 7, 0.5)
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