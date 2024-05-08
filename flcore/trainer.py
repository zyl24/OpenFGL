import torch
import random
from data.distributed_dataset_loader import FGLDataset
from utils.basic_utils import load_client, load_server

class FGLTrainer:
    
    def __init__(self, args):
        self.args = args
        self.message_pool = {}
        fgl_dataset = FGLDataset(args)
        self.device = torch.device(f"cuda:{args.gpuid}" if (torch.cuda.is_available() and args.use_cuda) else "cpu")
        self.clients = [load_client(args, client_id, fgl_dataset.local_data[client_id], fgl_dataset.processed_dir, self.message_pool, self.device) for client_id in range(self.args.num_clients)]
        self.server = load_server(args, fgl_dataset.global_data, fgl_dataset.processed_dir, self.message_pool, self.device)
        
        self.best_val_acc = 0
        self.best_test_acc = 0
        self.best_round = 0
        
    def train(self):
        
        for round_id in range(self.args.num_rounds):
            sampled_clients = sorted(random.sample(list(range(self.args.num_clients)), int(self.args.num_clients * self.args.client_frac)))
            print(f"sampled_clients: {sampled_clients}")
            self.message_pool["round"] = round_id
            self.message_pool["sampled_clients"] = sampled_clients
            self.server.send_message()
            for client_id in sampled_clients:
                self.clients[client_id].execute()
                self.clients[client_id].send_message()
            self.server.execute()
            
        
            
            # if self.args.evaluation_mode == "local_model_on_local_data":
            #     self.eval_local_model_on_local_data()
            # elif self.args.evaluation_mode == "local_model_on_global_data":
            #     self.eval_local_model_on_global_data()
            # elif self.args.evaluation_mode == "global_model_on_global_data":
            #     self.eval_global_model_on_global_data()
            # elif self.args.evaluation_mode == "global_model_on_local_data":
            #     self.eval_global_model_on_local_data()
            # else:
            #     raise ValueError
            
            
    def eval_global_model_on_global_data(self):
        if self.server.personalized:
            global_val_acc = 0
            global_test_acc = 0
            
            for client_id in range(self.args.num_clients):
                self.server.switch_personzlied_global_model(client_id)
                result = self.server.task.evaluate()
                global_val_acc += result["accuracy_val"]
                global_test_acc += result["accuracy_test"]
                
            global_val_acc /= self.args.num_clients
            global_test_acc /= self.args.num_clients
        else:
            result = self.server.task.evaluate()
            global_val_acc, global_test_acc = result["accuracy_val"], result["accuracy_test"]
        
        if global_val_acc > self.best_val_acc:
            self.best_val_acc = global_val_acc
            self.best_test_acc = global_test_acc
            self.best_round = self.message_pool["round"]
        
        print(f"curr_round: {self.message_pool['round']}\tcurr_val: {global_val_acc:.4f}\tcurr_test: {global_test_acc:.4f}")
        print(f"best_round: {self.best_round}\tbest_val: {self.best_val_acc:.4f}\tbest_test: {self.best_test_acc:.4f}")
        print("-"*50)
            

        
        
        
    def eval_local_model_on_local_data(self):
        # download -> local-train -> evaluate on local data
        global_val_acc = 0
        global_test_acc = 0
        tot_nodes = 0
        
        for client_id in range(self.args.num_clients):
            result = self.clients[client_id].task.evaluate()
            val_acc, test_acc = result["accuracy_val"], result["accuracy_test"]
            num_nodes = self.clients[client_id].task.num_samples
            global_val_acc += val_acc * num_nodes
            global_test_acc += test_acc * num_nodes
            tot_nodes += num_nodes
        global_val_acc /= tot_nodes
        global_test_acc /= tot_nodes
        
        if global_val_acc > self.best_val_acc:
            self.best_val_acc = global_val_acc
            self.best_test_acc = global_test_acc
            self.best_round = self.message_pool["round"]
        
        print(f"curr_round: {self.message_pool['round']}\tcurr_val: {global_val_acc:.4f}\tcurr_test: {global_test_acc:.4f}")
        print(f"best_round: {self.best_round}\tbest_val: {self.best_val_acc:.4f}\tbest_test: {self.best_test_acc:.4f}")
        print("-"*50)
        
        
        
    def eval_local_model_on_global_data(self):
        # download -> local-train -> evaluate on global data
        global_val_acc = 0
        global_test_acc = 0
        tot_nodes = 0
        
        for client_id in range(self.args.num_clients):
            result = self.clients[client_id].task.evaluate(self.server.task.splitted_data)
            val_acc, test_acc = result["accuracy_val"], result["accuracy_test"]
            num_nodes = self.clients[client_id].task.num_samples
            global_val_acc += val_acc * num_nodes
            global_test_acc += test_acc * num_nodes
            tot_nodes += num_nodes
        global_val_acc /= tot_nodes
        global_test_acc /= tot_nodes
        
        if global_val_acc > self.best_val_acc:
            self.best_val_acc = global_val_acc
            self.best_test_acc = global_test_acc
            self.best_round = self.message_pool["round"]
        
        print(f"curr_round: {self.message_pool['round']}\tcurr_val: {global_val_acc:.4f}\tcurr_test: {global_test_acc:.4f}")
        print(f"best_round: {self.best_round}\tbest_val: {self.best_val_acc:.4f}\tbest_test: {self.best_test_acc:.4f}")
        print("-"*50)
            
    
    
    def eval_global_model_on_local_data(self):
        # for non-personalized fl-algorithm:
        # server: global model -> evaluate on local data
        # client: download -> local-train -> evaluate on local data
    
        # for personalized fl-algorithm:
        # server: personalized global model -> evaluate on local data
        # client: download -> evaluate on local data
        global_val_acc = 0
        global_test_acc = 0
        tot_nodes = 0
        
        for client_id in range(self.args.num_clients):
            if self.server.personalized:
                self.server.switch_personalized_global_model(client_id)
                
            result = self.server.task.evaluate(self.clients[client_id].task.splitted_data)
            val_acc, test_acc = result["accuracy_val"], result["accuracy_test"]
            num_nodes = self.clients[client_id].task.num_samples
            global_val_acc += val_acc * num_nodes
            global_test_acc += test_acc * num_nodes
            tot_nodes += num_nodes
        global_val_acc /= tot_nodes
        global_test_acc /= tot_nodes
        
        if global_val_acc > self.best_val_acc:
            self.best_val_acc = global_val_acc
            self.best_test_acc = global_test_acc
            self.best_round = self.message_pool["round"]
        
        print(f"curr_round: {self.message_pool['round']}\tcurr_val: {global_val_acc:.4f}\tcurr_test: {global_test_acc:.4f}")
        print(f"best_round: {self.best_round}\tbest_val: {self.best_val_acc:.4f}\tbest_test: {self.best_test_acc:.4f}")
        print("-"*50)
            
    
    
    
    
    
        
        
        
    
        
        
        