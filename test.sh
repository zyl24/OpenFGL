python main.py --root ./dataset --dataset Cora --processing random_feature_noise --model gcn --metric accuracy

python main.py --root ./dataset --dataset Cora --processing edge_random_mask --model gcn --metric accuracy

python main.py --root ./dataset --dataset Cora --processing label_noise --model gcn --metric accuracy

python main.py --root ./dataset --dataset Cora --processing label_sparsity --model gcn --metric accuracy