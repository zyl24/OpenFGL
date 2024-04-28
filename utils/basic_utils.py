import torch
import random
import numpy as np

def check_args(args):
    pass


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    

    
    
def load_client(args, client_id, data, data_dir, message_pool, device):
    if args.fl_algorithm == "fedavg":
        from flcore.fedavg.client import FedAvgClient
        return FedAvgClient(args, client_id, data, data_dir, message_pool, device)
    elif args.fl_algorithm == "fedprox":
        from flcore.fedprox.client import FedProxClient
        return FedProxClient(args, client_id, data, data_dir, message_pool, device)
    elif args.fl_algorithm == "scaffold":
        from flcore.scaffold.client import ScaffoldClient
        return ScaffoldClient(args, client_id, data, data_dir, message_pool, device)
    elif args.fl_algorithm == "moon":
        from flcore.moon.client import MoonClient
        return MoonClient(args, client_id, data, data_dir, message_pool, device)
    elif args.fl_algorithm == "feddc":
        from flcore.feddc.client import FedDCClient
        return FedDCClient(args, client_id, data, data_dir, message_pool, device)
    elif args.fl_algorithm == "fedproto":
        from flcore.fedproto.client import FedProtoClient
        return FedProtoClient(args, client_id, data, data_dir, message_pool, device)
    elif args.fl_algorithm == "fedtgp":
        from flcore.fedtgp.client import FedTGPClient
        return FedTGPClient(args, client_id, data, data_dir, message_pool, device)
    elif args.fl_algorithm == "fedpub":
        from flcore.fedpub.client import FedPubClient
        return FedPubClient(args, client_id, data, data_dir, message_pool, device)
    elif args.fl_algorithm == "fedstar":
        from flcore.fedstar.client import FedStarClient
        return FedStarClient(args, client_id, data, data_dir, message_pool, device)
    
def load_server(args, global_data, data_dir, message_pool, device):
    if args.fl_algorithm == "fedavg":
        from flcore.fedavg.server import FedAvgServer
        return FedAvgServer(args, global_data, data_dir, message_pool, device)
    elif args.fl_algorithm == "fedprox":
        from flcore.fedprox.server import FedProxServer
        return FedProxServer(args, global_data, data_dir, message_pool, device)
    elif args.fl_algorithm == "scaffold":
        from flcore.scaffold.server import ScaffoldServer
        return ScaffoldServer(args, global_data, data_dir, message_pool, device)
    elif args.fl_algorithm == "moon":
        from flcore.moon.server import MoonServer
        return MoonServer(args, global_data, data_dir, message_pool, device)
    elif args.fl_algorithm == "feddc":
        from flcore.feddc.server import FedDCServer
        return FedDCServer(args, global_data, data_dir, message_pool, device)
    elif args.fl_algorithm == "fedproto":
        from flcore.fedproto.server import FedProtoServer
        return FedProtoServer(args, global_data, data_dir, message_pool, device)
    elif args.fl_algorithm == "fedtgp":
        from flcore.fedtgp.server import FedTGPServer
        return FedTGPServer(args, global_data, data_dir, message_pool, device)
    elif args.fl_algorithm == "fedpub":
        from flcore.fedpub.server import FedPubServer
        return FedPubServer(args, global_data, data_dir, message_pool, device)
    elif args.fl_algorithm == "fedstar":
        from flcore.fedstar.server import FedStarServer
        return FedStarServer(args, global_data, data_dir, message_pool, device)
    
    
def load_optim(args):
    if args.optim == "adam":
        from torch.optim import Adam
        return Adam
    
    
def load_task(args, client_id, data, data_dir, device):
    if args.task == "node_cls":
        from task.node_cls import NodeClsTask
        return NodeClsTask(args, client_id, data, data_dir, device)
    elif args.task == "graph_cls":
        from task.graph_cls import GraphClsTask
        return GraphClsTask(args, client_id, data, data_dir, device)
    


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
    
    
    