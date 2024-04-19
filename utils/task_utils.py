def load_node_cls_default_model(args, input_dim, output_dim, client_id=None):
    if client_id is not None and len(args.model) > 1:
        model_id = int(len(args.model) * args.client_id / args.num_clients)
        model_name = args.model[model_id]
    else:
        model_name = args.model[0]
    
    if model_name == "gcn":
        from model.gcn import GCN
        return GCN(input_dim=input_dim, hid_dim=args.hid_dim, output_dim=output_dim, dropout=args.dropout)
    elif model_name == "gat":
        from model.gat import GAT
        return GAT(input_dim=input_dim, hid_dim=args.hid_dim, output_dim=output_dim, dropout=args.dropout)
    elif model_name == "graphsage":
        from model.graphsage import GraphSAGE
        return GraphSAGE(input_dim=input_dim, hid_dim=args.hid_dim, output_dim=output_dim, dropout=args.dropout)
    elif model_name == "sgc":
        from model.sgc import SGC
        return SGC(input_dim=input_dim, hid_dim=args.hid_dim, output_dim=output_dim, dropout=args.dropout)
    elif model_name == "gcn2":
        from model.gcn2 import GCN2
        return GCN2(input_dim=input_dim, hid_dim=args.hid_dim, output_dim=output_dim, dropout=args.dropout)
    else:
        raise ValueError

