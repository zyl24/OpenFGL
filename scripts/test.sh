cd /home/ai2/work/OPENFGL

datasets=("Cora")
fl_algorithms=("fedavg" "fedprox" "scaffold" "moon"  "feddc" "fedgta" "fedproto" "fedtgp" "adafgl" "fedpub" "fedtad" "fedsage_plus")
simulation_modes=("fedsubgraph_label_dirichlet" "fedsubgraph_louvain" "fedsubgraph_louvain_clustering" "fedsubgraph_metis" "fedsubgraph_metis_clustering")
num_clients=("10")
repeats=1


for dataset in "${datasets[@]}"; do
    for simulation_mode in "${simulation_modes[@]}"; do
        for fl_algorithm in "${fl_algorithms[@]}"; do
            for num_client in "${num_clients[@]}"; do
                for (( i=1; i<=repeats; i++ )); do
                    echo "Running python main.py with dataset=${dataset}, fl_algorithm=${fl_algorithm}, num_clients=${num_client}, repeat=${i}"
                    python main.py --seed 0 --dataset "${dataset}" --fl_algorithm "${fl_algorithm}" --num_clients "${num_client}" --scenario fedsubgraph --processing raw --num_rounds 100 --num_epochs 3 --simulation_mode "${simulation_mode}" --task node_cls --lr 1e-2 --model gcn --metrics accuracy > /dev/null 2>&1
                done
            done
        done
    done
done