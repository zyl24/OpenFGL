import os
import pickle
import numpy as np
import matplotlib.pyplot as plt





def create_variation(base, noise_level=1, decay=0.2, seed=None):
    np.random.seed(seed)
    # Ensure noise_level is never negative
    effective_noise_level = max(noise_level, 0) 
    return np.clip(base + np.random.normal(0, effective_noise_level, size=x.shape) - decay * np.arange(len(x)), 0, 80)

def create_data(search_str):
    dataset = ["Cora", "CiteSeer", "PubMed"]
    simu = ["label_dirichlet_10.00", "louvain_1", "metis_clustering_100", "louvain_clustering_1", "metis"]
    
    
    path = f"/home/ai2/work/dataset/distrib/fedsubgraph_{simu[4]}_{dataset[0]}_client_10/debug/"
    search_str = search_str
    found_files = []
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            if search_str in filename:
                full_path = os.path.join(dirpath, filename)
                found_files.append(full_path)
    
    total_result = []  
      
    for found_name in found_files:
        print(found_name)
        with open(found_name, 'rb') as file:
            log_file = pickle.load(file)
        # print(log_file["time"])
        # print(log_file["metric"][-1])
        total_result.append([metric["current_test_accuracy"] for metric in log_file["metric"]])
    
    
    num_result = len(total_result)
    tot_result_np = np.array(total_result)
    base_line = tot_result_np.mean(axis=0)
    lower_bound = tot_result_np.min(axis=0)
    upper_bound = tot_result_np.max(axis=0)

    
    
    
    
    
    return base_line, lower_bound, upper_bound

# Data generation with confidence intervals
fl_algorithms=("fedavg" "fedprox" "scaffold" "moon"  "feddc" "fedgta" "fedproto" "fedtgp" "adafgl" "fedpub" "fedsage_plus") # fgssl, fedgl, fggp, feddep
algorithms = {
    "FedAvg": create_data("fedavg"),
    "FedProx": create_data("fedprox"),
    "Scaffold": create_data("scaffold"),
    "MOON": create_data("moon"),
    "FedDC": create_data("feddc"),
    "FedTGP": create_data("fedtgp"),
    "FedProto": create_data("fedproto"),
    "FedSage+": create_data("fedsage_plus"),
    "FedPub": create_data("fedpub"),
    "FedGTA": create_data("fedgta"),
    "AdaFGL": create_data("adafgl")
}

# Plotting
plt.figure(figsize=(10, 6))
for label, (values, lower, upper) in algorithms.items():
    print(label, f"{values[-1]}")
    plt.plot(range(len(values)), values, label=label)
    plt.fill_between(range(len(lower)), lower, upper, alpha=0.2)

plt.title('Test Accuracy vs Running Time')
plt.xlabel('Running Time (s)')
plt.ylabel('Test Accuracy (%)')
plt.legend()
plt.grid(True)
plt.savefig("/home/ai2/work/OPENFGL/fig.png")
