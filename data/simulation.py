import os
import random
import torch

def extract_floats(s):
    from decimal import Decimal
    parts = s.split('-')
    train = float(parts[0])
    val = float(parts[1])
    test = float(parts[2])
    assert Decimal(parts[0]) + Decimal(parts[1]) + Decimal(parts[2]) == Decimal(1)
    return train, val, test

def idx_to_mask_tensor(idx_list, length):
    mask = torch.zeros(length)
    mask[idx_list] = 1
    return mask

def fedgraph_simulation(args, global_dataset, shuffle=True):
    if type(global_dataset) is list:
        print("Conducting Fed-graph Simulation across Multiple Datasets.")
        
        
    else:
        print("Conducting Fed-graph Simulation within Single Dataset.")
        save_dir = os.path.join(args.root, "fedgraph", args.dataset[0], "simulation", f"{args.num_clients}_clients_{args.train_val_test}")
        num_graphs = len(global_dataset)
        shuffled_graph_idx = list(range(num_graphs))
        random.shuffle(shuffled_graph_idx)
        os.makedirs(save_dir, exist_ok=True)
        for client_id in range(args.num_clients):
            local_graphs = global_dataset[shuffled_graph_idx[num_graphs * client_id // args.num_clients : num_graphs * (client_id+1) // args.num_clients]]
            train_, val_, test_ = extract_floats(args.train_val_test)
            num_local_graphs = len(local_graphs)
            
            if local_graphs[0].y.dtype is torch.int64: # label -> torch.int64, for classification problem
                num_classes = global_dataset.num_classes
                local_graphs.train_mask = idx_to_mask_tensor([], num_local_graphs)
                local_graphs.val_mask = idx_to_mask_tensor([], num_local_graphs)
                local_graphs.test_mask = idx_to_mask_tensor([], num_local_graphs)
                for class_i in range(num_classes):
                    class_i_local_graphs_mask = local_graphs.y == class_i
                    num_class_i_local_graphs = class_i_local_graphs_mask.sum()
                    local_graphs.train_mask += idx_to_mask_tensor(class_i_local_graphs_mask.nonzero().squeeze().tolist()[:int(train_ * num_class_i_local_graphs)], num_local_graphs)
                    local_graphs.val_mask += idx_to_mask_tensor(class_i_local_graphs_mask.nonzero().squeeze().tolist()[int(train_ * num_class_i_local_graphs) : int((train_+val_) * num_class_i_local_graphs)], num_local_graphs)
                    local_graphs.test_mask += idx_to_mask_tensor(class_i_local_graphs_mask.nonzero().squeeze().tolist()[int((train_+val_) * num_class_i_local_graphs): ], num_local_graphs)
                assert (local_graphs.train_mask + local_graphs.val_mask + local_graphs.test_mask).sum() == num_local_graphs
            else: # label -> torch.float64, for regression problem
                local_graphs.train_mask = idx_to_mask_tensor(range(int(train_ * num_local_graphs)), num_local_graphs)
                local_graphs.val_mask = idx_to_mask_tensor(range(int(train_ * num_local_graphs), int((train_+val_) * num_local_graphs)), num_local_graphs)
                local_graphs.test_mask = idx_to_mask_tensor(range(int((train_+val_) * num_local_graphs), num_local_graphs), num_local_graphs)
                assert (local_graphs.train_mask + local_graphs.val_mask + local_graphs.test_mask).sum() == num_local_graphs
            torch.save(local_graphs, os.path.join(save_dir, f"client_{client_id}.pt"))
        
        
        
        
        
        
        
    
    