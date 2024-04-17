import torch
import random
import numpy as np


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    
    
def load_node_cls_default_model(args, input_dim, output_dim, client_id=None):
    if client_id is not None and len(args.model) > 1:
        model_id = int(len(args.model) * args.client_id / args.num_clients)
        model_name = args.model[model_id]
    else:
        model_name = args.model[0]
    
    if model_name == "gcn":
        from model.gcn import GCN
        return GCN(input_dim=input_dim, hid_dim=args.hid_dim, output_dim=output_dim, dropout=args.dropout)
    
    
def load_client(args, client_id, data, data_dir, message_pool, device):
    if args.fl_algorithm == "fedavg":
        from flcore.fedavg.client import FedAvgClient
        return FedAvgClient(args, client_id, data, data_dir, message_pool, device)
    elif args.fl_algorithm == "fedprox":
        from flcore.fedprox.client import FedProxClient
        return FedProxClient(args, client_id, data, data_dir, message_pool, device)


def load_server(args, global_data, data_dir, message_pool, device):
    if args.fl_algorithm == "fedavg":
        from flcore.fedavg.server import FedAvgServer
        return FedAvgServer(args, global_data, data_dir, message_pool, device)
    elif args.fl_algorithm == "fedprox":
        from flcore.fedprox.server import FedProxServer
        return FedProxServer(args, global_data, data_dir, message_pool, device)
    
def load_optim(args):
    if args.optim == "adam":
        from torch.optim import Adam
        return Adam
    
    
def load_task(args, client_id, data, data_dir, device, custom_model=None):
    if args.task == "node_cls":
        from task.fedsubgraph.node_cls import NodeClsTask
        return NodeClsTask(args, client_id, data, data_dir, device, custom_model)
    


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



def mask_tensor_to_idx(tensor):
    result = tensor.nonzero().squeeze().tolist()
    if type(result) is not list:
        result = [result]
    return result
    
    
    