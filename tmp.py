import pickle
path = "/home/ai2/work/OPENFGL/exp/effectiveness_q1/subgraph_fl/fedavg.pkl"


with open(path, 'rb') as file:
    log_file = pickle.load(file)
    for key, val in log_file["args"].items():
        print("{0:20}\t{1:20}".format(key, f'{val}'))
    print(log_file["time"])
    # print(log_file["metric"]) # len=num_rounds
    
    