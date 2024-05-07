from config import args
from flcore.trainer import FGLTrainer
from utils.basic_utils import seed_everything
print(args)

# args.scenario = "fedgraph"
# args.simulation_mode = "fedgraph_label_dirichlet"
# args.task = "graph_cls"
# args.model = ["gin"]
# args.dataset = ["COX"]
# args.fl_algorithm = "fedavg"
# args.num_clients = 3
# args.evaluation_mode = "global_model_on_local_data" 

args.scenario = "fedsubgraph"
args.simulation_mode = "fedsubgraph_louvain"
args.task = "node_cls"
args.model = ["gcn"]
args.dataset = ["Cora"]
args.num_clients = 7
args.fl_algorithm = "fedsage_plus"
args.evaluation_mode = "global_model_on_local_data" 



seed_everything(args.seed)
trainer = FGLTrainer(args)
trainer.train()

"""
Coar - Louvain - GCN - 10 Clients

> fedavg    -   69.27
> fedprox   -   71.26
> scaffold  -   80.65
> moon      -   70.71
> feddc     -
> fedsage   -
> fedpub    -   73.79
> fedgta    -   72.16
> fedtad    -   72.08
    
    
"""