import os
import pickle


for root, dirs, files in os.walk("/home/ai2/work/OPENFGL/exp/robustness_q3/subgraph_fl/Cora"):
    for file in files:
        if "fedsage" in file and "feature" in file:
            print(os.path.join(root, file))
            with open(os.path.join(root, file), 'rb') as file:
                log_file = pickle.load(file)
                # for key, val in log_file["args"].items():
                #     print("{0:20}\t{1:20}".format(key, f'{val}'))
                # print(log_file["time"])
                
                best_val = 0
                best_test = 0
                for i in range(len(log_file["metric"])):
                    if log_file["metric"][i]["current_val_accuracy"] > best_val:
                        best_val = log_file["metric"][i]["current_val_accuracy"]
                        best_test = log_file["metric"][i]["current_test_accuracy"]
                
                print("best_val {:.4f}".format(best_val)) 
                print("best_test {:.4f}".format(best_test)) 
    
    