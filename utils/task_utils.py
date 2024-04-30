from torch_geometric.nn.pool import *

def load_graph_cls_default_model(args, input_dim, output_dim, client_id=None):
    if client_id is None: # server
        if len(args.model) > 1:
            return None
        else:
            model_name = args.model[0]
    else: # client
        if len(args.model) > 1:
            model_id = int(len(args.model) * client_id / args.num_clients)
            model_name = args.model[model_id]
        else:
            model_name = args.model[0]
        
            
    if model_name == "gin":
        from model.gin import GIN
        return GIN(input_dim=input_dim, hid_dim=args.hid_dim, output_dim=output_dim, num_layers=args.num_layers, dropout=args.dropout)
    else:
        raise ValueError



def load_node_cls_default_model(args, input_dim, output_dim, client_id=None):
    if client_id is None: # server
        if len(args.model) > 1:
            return None
        else:
            model_name = args.model[0]
    else: # client
        if len(args.model) > 1:
            model_id = int(len(args.model) * client_id / args.num_clients)
            model_name = args.model[model_id]
        else:
            model_name = args.model[0]
    
    if model_name == "gcn":
        from model.gcn import GCN
        return GCN(input_dim=input_dim, hid_dim=args.hid_dim, output_dim=output_dim, num_layers=args.num_layers, dropout=args.dropout)
    elif model_name == "gat":
        from model.gat import GAT
        return GAT(input_dim=input_dim, hid_dim=args.hid_dim, output_dim=output_dim, num_layers=args.num_layers, dropout=args.dropout)
    elif model_name == "graphsage":
        from model.graphsage import GraphSAGE
        return GraphSAGE(input_dim=input_dim, hid_dim=args.hid_dim, output_dim=output_dim, num_layers=args.num_layers, dropout=args.dropout)
    elif model_name == "sgc":
        from model.sgc import SGC
        return SGC(input_dim=input_dim, hid_dim=args.hid_dim, output_dim=output_dim, num_layers=args.num_layers, dropout=args.dropout)
    elif model_name == "gcn2":
        from model.gcn2 import GCN2
        return GCN2(input_dim=input_dim, hid_dim=args.hid_dim, output_dim=output_dim, num_layers=args.num_layers, dropout=args.dropout)
    else:
        raise ValueError

