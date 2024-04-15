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


def load_client(args):
    
    if args.fl_algorithm == "fedavg":
        from flcore.fedavg.client import FedAvgClient as Client
        from flcore.fedavg.server import FedAvgServer as Server
    
    
    
    