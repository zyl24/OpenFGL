import torch



    
def load_node_cls_default_model(args, input_dim, output_dim, client_id=None):
    if client_id is not None and len(args.model) > 1:
        model_id = int(len(args.model) * args.client_id / args.num_clients)
        model_name = args.model[model_id]
    else:
        model_name = args.model[0]
    
    if model_name == "gcn":
        from model.gcn import GCN
        return GCN(input_dim=input_dim, output_dim=output_dim, dropout=args.dropout)
    
    
def load_client(args, client_id, data, data_dir, message_pool):
    if args.fl_algorithm == "fedavg":
        from flcore.fedavg.client import FedAvgClient
        return FedAvgClient(args, client_id, data, data_dir, message_pool)



def load_server(args, global_data, data_dir, message_pool):
    if args.fl_algorithm == "fedavg":
        from flcore.fedavg.server import FedAvgServer
        return FedAvgServer(args, global_data, data_dir, message_pool)
    
def load_optim(args):
    if args.optim == "adam":
        from torch.optim import Adam
        return Adam
    
    
def load_task(args, client_id, data, data_dir, custom_model=None, custom_optim=None, custom_loss_fn=None):
    if args.task == "node_cls":
        from task.fedsubgraph.node_cls import NodeClsTask
        return NodeClsTask(args, client_id, data, data_dir, custom_model, custom_optim, custom_loss_fn)
    


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



    
    
    