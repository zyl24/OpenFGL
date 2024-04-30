from config import args
from flcore.trainer import FGLTrainer
from utils.basic_utils import seed_everything
print(args)


# args.dataset = ["Cora"]
# args.scenario = "fedsubgraph"
# args.task = "node_cls"
# args.simulation_mode = "fedsubgraph_louvain_clustering"


seed_everything(args.seed)
trainer = FGLTrainer(args)
trainer.train()
