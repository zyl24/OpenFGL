import os
import copy
import pickle
import datetime
import time


class Logger:
    
    def __init__(self, args, message_pool, task_path):
        self.args = args
        self.message_pool = message_pool
        self.debug = self.args.debug
        self.task_path = task_path
        self.metrics_list = []
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_path = os.path.join(self.task_path, "debug", f"{self.args.fl_algorithm}_{current_time}.pkl")
        self.start_time = time.time()
    
    def add_log(self, evaluation_result):
        if not self.debug:
            return
        self.metrics_list.append(copy.deepcopy(evaluation_result))
    
    def save(self):
        if not self.debug:
            return
        
        if not os.path.exists(os.path.dirname(self.log_path)):
            os.makedirs(os.path.dirname(self.log_path))
            
            
        log = {
            "args": vars(self.args),
            "time": time.time() - self.start_time,
            "metric": self.metrics_list
        }
        with open(self.log_path, 'wb') as file:
            pickle.dump(log, file)