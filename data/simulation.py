import copy
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


def local_graphs_train_val_test_split(local_graphs, split, num_classes=None):
    num_local_graphs = len(local_graphs)
    train_, val_, test_ = extract_floats(split)
    
    if num_classes is not None: # for classification problem
        local_graphs.train_mask = idx_to_mask_tensor([], num_local_graphs)
        local_graphs.val_mask = idx_to_mask_tensor([], num_local_graphs)
        local_graphs.test_mask = idx_to_mask_tensor([], num_local_graphs)
        for class_i in range(num_classes):
            class_i_local_graphs_mask = local_graphs.y == class_i
            num_class_i_local_graphs = class_i_local_graphs_mask.sum()
            local_graphs.train_mask += idx_to_mask_tensor(class_i_local_graphs_mask.nonzero().squeeze().tolist()[:int(train_ * num_class_i_local_graphs)], num_local_graphs)
            local_graphs.val_mask += idx_to_mask_tensor(class_i_local_graphs_mask.nonzero().squeeze().tolist()[int(train_ * num_class_i_local_graphs) : int((train_+val_) * num_class_i_local_graphs)], num_local_graphs)
            local_graphs.test_mask += idx_to_mask_tensor(class_i_local_graphs_mask.nonzero().squeeze().tolist()[int((train_+val_) * num_class_i_local_graphs): ], num_local_graphs)
    else: # for regression problem
        local_graphs.train_mask = idx_to_mask_tensor(range(int(train_ * num_local_graphs)), num_local_graphs)
        local_graphs.val_mask = idx_to_mask_tensor(range(int(train_ * num_local_graphs), int((train_+val_) * num_local_graphs)), num_local_graphs)
        local_graphs.test_mask = idx_to_mask_tensor(range(int((train_+val_) * num_local_graphs), num_local_graphs), num_local_graphs)
    
    assert (local_graphs.train_mask + local_graphs.val_mask + local_graphs.test_mask).sum() == num_local_graphs

    
def fedgraph_simulation(args, global_dataset, shuffle=True):
    if type(global_dataset) is list:
        print("Conducting fed-graph simulation across multiple datasets.")
        fmt_list = copy.deepcopy(args.dataset)
        fmt_list = sorted(fmt_list)
        
        save_dir = os.path.join(args.root, "fedgraph", f"multiple_{'_'.join(fmt_list)}", "simulation", f"{args.num_clients}_clients_{args.train_val_test}")
        assert len(global_dataset) == args.num_clients , f"For fed-graph simulation across multiple datasets, number of clients must be equal to the number of used datasets. (args.num_clients={args.num_clients}; used_datasets: {args.dataset})"
        os.makedirs(save_dir, exist_ok=True)
        for client_id in range(args.num_clients):
            local_graphs = global_dataset[client_id]
            num_graphs = len(local_graphs)
            shuffled_graph_idx = list(range(num_graphs))
            if shuffle:
                random.shuffle(shuffled_graph_idx)
            local_graphs_train_val_test_split(local_graphs[shuffled_graph_idx], args.train_val_test, num_classes=local_graphs.num_classes if local_graphs[0].y.dtype is torch.int64 else None)
            torch.save(local_graphs, os.path.join(save_dir, f"client_{client_id}.pt"))
    else:
        print("Conducting fed-graph simulation within single dataset.")
        save_dir = os.path.join(args.root, "fedgraph", args.dataset[0], "simulation", f"{args.num_clients}_clients_{args.train_val_test}")
        os.makedirs(save_dir, exist_ok=True)
        num_graphs = len(global_dataset)
        shuffled_graph_idx = list(range(num_graphs))
        if shuffle:
            random.shuffle(shuffled_graph_idx)
        for client_id in range(args.num_clients):
            local_graphs = global_dataset[shuffled_graph_idx[num_graphs * client_id // args.num_clients : num_graphs * (client_id+1) // args.num_clients]]
            local_graphs_train_val_test_split(local_graphs, args.train_val_test, num_classes=global_dataset.num_classes if local_graphs[0].y.dtype is torch.int64 else None)
            torch.save(local_graphs, os.path.join(save_dir, f"client_{client_id}.pt"))
        
        
        
        
        
        
        
    
    