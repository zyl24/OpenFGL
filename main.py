from config import args
from flcore.trainer import FGLTrainer
from utils.basic_utils import seed_everything

args.fl_algorithm = "fedtgp"
args.model = ["gcn", "gat", "graphsage"]
seed_everything(args.seed)
trainer = FGLTrainer(args)
trainer.train()

# fedavg    73.42
# fedprox   74.48
# scaffold  75.69
# moon      75.59
# feddc     69.92
# fedproto  67.54
# fedtgp    66.65

