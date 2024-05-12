import pickle


path = "/home/ai2/work/dataset/distrib/fedsubgraph_louvain_1_Cora_client_10/debug/fedavg_2024-05-12_23-48-14.pkl"


with open(path, 'rb') as file:
    log_file = pickle.load(file)
    
    
print(log_file["time"])
print(log_file["metric"])

    