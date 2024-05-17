cd /home/ai2/work/OPENFGL

datasets=("COX2")
fl_algorithms=("fedavg" "fedprox" "scaffold" "moon"  "feddc" "gcfl_plus" "fedstar")
simulation_modes=("fedgraph_label_dirichlet")
num_clients=("5")
repeats=1


for dataset in "${datasets[@]}"; do
    for simulation_mode in "${simulation_modes[@]}"; do
        for fl_algorithm in "${fl_algorithms[@]}"; do
            for num_client in "${num_clients[@]}"; do
                for (( i=1; i<=repeats; i++ )); do
                    echo "Running python main.py with dataset=${dataset}, fl_algorithm=${fl_algorithm}, num_clients=${num_client}, repeat=${i}"
                    python main.py --seed 0 --dataset "${dataset}" --fl_algorithm "${fl_algorithm}" --num_clients "${num_client}" --scenario fedgraph --processing raw --num_rounds 100 --num_epochs 3 --simulation_mode "${simulation_mode}" --task graph_cls --lr 1e-2 --model gin --metrics accuracy > /dev/null 2>&1
                done
            done
        done
    done
done



python main.py --seed 0 --dataset COX2 --fl_algorithm fedavg --num_clients 5 --scenario fedgraph --processing raw --num_rounds 100 --num_epochs 3 --simulation_mode fedgraph_label_dirichlet --task graph_cls --lr 1e-2 --model gin --metrics accuracy