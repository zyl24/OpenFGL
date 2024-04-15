import random
from data.distributed_dataset_loader import FGLDataset
from flcore.fedavg.client import FedAvgClient
from flcore.fedavg.server import FedAvgServer


class FGLTrainer:
    
    def __init__(self, args):
        self.args = args
        self.message_pool = {}
        fgl_dataset = FGLDataset(args)
        self.clients = [FedAvgClient(client_id, message_pool=self.message_pool, args=args, data=fgl_dataset.local_data[client_id]) for client_id in range(self.args.num_clients)]
        self.server = FedAvgServer(message_pool=self.message_pool)
    
    def train(self):
        best_val_acc = 0
        best_test_acc = 0
        best_round = 0


        for round_id in range(self.args.num_rounds):
            sampled_clients = random.sample(list(range(self.args.num_clients)), int(self.args.num_clients * self.client_frac))
            print(f"sampled_clients: {sampled_clients}")
            self.message_pool["round"] = round_id
            self.message_pool["sampled_clients"] = sampled_clients
            self.server.send_message()
            for client_id in sampled_clients:
                self.clients[client_id].receive_message()
                self.clients[client_id].execute()
                self.clients[client_id].send_message()
            self.server.receive_message()
            self.server.execute()
            
            # evaluation
            global_val_acc = 0
            global_test_acc = 0
            tot_nodes = 0
            
            for client_id in range(self.args.num_clients):
                result = self.clients[client_id].personalized_evaluate()
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
                
    

        
    
    
    
    
    
        
        
        