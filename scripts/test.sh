cd /home/ai2/work/OPENFGL

datasets=("Cora" "CiteSeer" "PubMed")
fl_algorithms=("fedavg" "fedprox" "scaffold" "moon" "fedgta" "fedpub")
num_clients=("5" "10")
repeats=3


for dataset in "${datasets[@]}"; do
    for fl_algorithm in "${fl_algorithms[@]}"; do
        for num_client in "${num_clients[@]}"; do
            for (( i=1; i<=repeats; i++ )); do
                echo "Running python main.py with dataset=${dataset}, fl_algorithm=${fl_algorithm}, num_clients=${num_client}, repeat=${i}"
                python main.py --dataset "${dataset}" --fl_algorithm "${fl_algorithm}" --num_clients "${num_client}" --scenario fedsubgraph --processing raw --num_rounds 100 --num_epochs 3 --simulation_mode fedsubgraph_louvain --task node_cls --lr 1e-2 --model gcn --metrics accuracy
            done
        done
    done
done