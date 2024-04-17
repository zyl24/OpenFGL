from config import args
from flcore.trainer import FGLTrainer


trainer = FGLTrainer(args)
trainer.train()
