from data.global_dataset_loader import load_global_dataset
import argparse
parser = argparse.ArgumentParser()



# global dataset settings 
parser.add_argument("--root", type=str, default="/home/ai2/work/dataset")
parser.add_argument("--scenairo", type=str, default="fedsubgraph")
parser.add_argument("--dataset", type=list, default=["Cora"])

# simulation settings
parser.add_argument("--num_clients", type=int, default=3)
# train/val/test
parser.add_argument("--train_val_test", type=str, default="0.8-0.1-0.1")


args = parser.parse_args()

if len(args.dataset) == 1:
    glb_dataset = load_global_dataset(args.root, args.scenairo, args.dataset[0])
else:
    glb_dataset = [load_global_dataset(args.root, args.scenairo, dataset) for dataset in args.dataset]

