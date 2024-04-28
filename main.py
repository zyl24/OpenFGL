from config import args
from flcore.trainer import FGLTrainer
from utils.basic_utils import seed_everything

seed_everything(args.seed)
trainer = FGLTrainer(args)
trainer.train()
