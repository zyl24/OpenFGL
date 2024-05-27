import pickle
path = "dataset/distrib/fedgraph_label_dirichlet_1.00_BZR_client_10/debug/fedavg_2024-05-24_14-43-03.pkl"



with open(path, 'rb') as file:
    log_file = pickle.load(file)
    print(log_file["args"])
    print(log_file["time"])
    print(log_file["metric"]) # len=num_rounds
    
    