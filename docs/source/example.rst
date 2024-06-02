Example
========================

.. code:: python

            from flcore.trainer import FGLTrainer
            from utils.basic_utils import seed_everything
            import sys
            
            sys.argv = ["main_test.py",
                        '--debug', 'False',
                        "--seed", '0',
                        '--scenario', 'fedsubgraph',
                        '--simulation_mode', 'fedsubgraph_louvain',
                        '--task', 'node_cls',
                        '--louvain_resolution', '1',
                        '--dataset', 'Cora',
                        '--model', 'gcn',
                        '--fl_algorithm', 'fedpub',
                        '--num_clients', '5',
                        '--num_epochs', '1',
                        '--metrics', 'accuracy']
            
            from config import args
            if args.seed != 0:
                seed_everything(args.seed)
                
            print(args)
            
            # set --root to store raw and processed dataset in your own path
            trainer = FGLTrainer(args)
            trainer.train()