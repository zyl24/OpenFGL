import random
from data.distributed_dataset_loader import FGLDataset
from utils.basic_utils import load_client, load_server

class FGLTrainer:
    
    def __init__(self, args):
        self.args = args
        self.message_pool = {}
        fgl_dataset = FGLDataset(args)
        self.clients = [load_client(args, client_id, fgl_dataset.local_data[client_id], fgl_dataset.processed_dir, self.message_pool) for client_id in range(self.args.num_clients)]
        self.server = load_server(args, fgl_dataset.global_data, fgl_dataset.processed_dir, self.message_pool)
    
    def train(self):
        
        for round_id in range(self.args.num_rounds):
            sampled_clients = random.sample(list(range(self.args.num_clients)), int(self.args.num_clients * self.args.client_frac))
            print(f"sampled_clients: {sampled_clients}")
            self.message_pool["round"] = round_id
            self.message_pool["sampled_clients"] = sampled_clients
            self.server.send_message()
            for client_id in sampled_clients:
                self.clients[client_id].execute()
                self.clients[client_id].send_message()
            self.server.execute()
            
            
    
    def personalized_evaluation(self, round_id):
        self.best_val_acc = 0
        self.best_test_acc = 0
        self.best_round = 0
        

        global_val_acc = 0
        global_test_acc = 0
        tot_nodes = 0
        
        for client_id in range(self.args.num_clients):
            result = self.clients[client_id].personalized_evaluate()
            val_acc, test_acc = result["val_accuracy"], result["test_accuracy"]
            num_nodes = self.clients[client_id].task.num_nodes
            global_val_acc += val_acc * num_nodes
            global_test_acc += test_acc * num_nodes
            tot_nodes += num_nodes
        global_val_acc /= tot_nodes
        global_test_acc /= tot_nodes
        
        if global_val_acc > self.best_val_acc:
            self.best_val_acc = global_val_acc
            self.best_test_acc = global_test_acc
            self.best_round = round_id
        
        print(f"best_round: {self.best_round}\tbest_val: {self.best_val_acc:.2f}\tbest_test: {self.best_test_acc:.2f}")
        print("-"*50)
            
    
    
    
    
    
        
        
        