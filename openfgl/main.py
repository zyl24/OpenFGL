from config import args
from flcore.trainer import FGLTrainer
from utils.basic_utils import seed_everything


if __name__=="__main__": 
    seed_everything(args.seed)
    fgltrainer = FGLTrainer(args = args)
    fgltrainer.train()
