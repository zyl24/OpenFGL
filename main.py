from data.global_dataset_loader import load_global_dataset
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--scenairo", type=str, default="fedgraph")
parser.add_argument("--root", type=str, default="/home/ai2/work/openfgl_data")
parser.add_argument("--dataset", type=str, default="Herg")

args = parser.parse_args()

print(load_global_dataset(args))

    