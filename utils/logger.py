import os
import copy
import pickle
import datetime



class Logger:
    
    def __init__(self, args, message_pool, task_path):
        self.args = args
        self.message_pool = message_pool
        self.debug = self.args.debug
        self.task_path = task_path
        self.log_list = []
        
    @property
    def log_path(self):
        return os.path.join(self.task_path, "debug", self.args.fl_algorithm, datetime.datetime.now())
    
    
    def add_log(self, evaluation_result):
        # 1. save task-oriented metric in each round
        self.log_list.append(copy.deepcopy(evaluation_result))
    
    def save(self):
        with open(self.log_path, 'wb') as file:
            pickle.dump(self.log_list, file)