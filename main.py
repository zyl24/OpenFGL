from config import args
from flcore.trainer import FGLTrainer
from utils.basic_utils import seed_everything

args.fl_algorithm = "moon"
seed_everything(args.seed)
trainer = FGLTrainer(args)
trainer.train()

# fedavg    73.42
# fedprox   74.57
# scaffold  75.69
