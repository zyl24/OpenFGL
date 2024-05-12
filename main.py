from config import args
from flcore.trainer import FGLTrainer
from utils.basic_utils import seed_everything
print(args)

args.scenario = "fedgraph"
args.simulation_mode = "fedgraph_label_dirichlet"
args.task = "graph_cls"
args.model = ["gin"]
args.dataset = ["COX2"]
args.fl_algorithm = "gcfl_plus"
args.num_clients = 3
args.evaluation_mode = "local_model_on_local_data" 

# args.scenario = "fedsubgraph"
# args.simulation_mode = "fedsubgraph_louvain"
# args.task = "node_clust"
# args.model = ["gcn"]
# args.dataset = ["Cora"]
# args.num_clients = 3
# args.num_epochs = 3
# args.fl_algorithm = "fedavg"
# args.evaluation_mode = "local_model_on_local_data" 
# args.num_rounds = 100
# # args.metrics = ["ap", "auc"]
# args.metrics = ["nmi", "ari", "clustering_accuracy"]
# args.lr = 1e-2


seed_everything(args.seed)
trainer = FGLTrainer(args)
trainer.train()

