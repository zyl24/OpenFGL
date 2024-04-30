from config import args
from flcore.trainer import FGLTrainer
from utils.basic_utils import seed_everything
print(args)

# args.scenario = "fedgraph"
# args.simulation_mode = "fedgraph_label_dirichlet"
# args.task = "graph_cls"
# args.model = ["gin"]
# args.dataset = ["COX2"]
# args.fl_algorithm = "fedavg"
# args.evaluation_mode = "personalized" 

args.scenario = "fedsubgraph"
args.simulation_mode = "fedsubgraph_label_dirichlet"
args.task = "node_cls"
args.model = ["gcn"]
args.dataset = ["Cora"]
args.fl_algorithm = "fedpub"
args.evaluation_mode = "personalized" 



seed_everything(args.seed)
trainer = FGLTrainer(args)
trainer.train()

