from data.global_dataset_loader import load_global_dataset
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--scenairo", type=str, default="fedgraph")
parser.add_argument("--root", type=str, default="/home/ai2/work/dataset")
parser.add_argument("--dataset", type=str, default="hERG") # torch.int64, torch.float64 

args = parser.parse_args()

glb_dataset = load_global_dataset(args)
print(glb_dataset)

print("ok")
    